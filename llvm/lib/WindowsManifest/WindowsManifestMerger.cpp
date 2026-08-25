//===-- WindowsManifestMerger.cpp ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file implements the .manifest merger class.
//
//===---------------------------------------------------------------------===//

#include "llvm/WindowsManifest/WindowsManifestMerger.h"
#include "llvm/Config/config.h"
#include "llvm/Support/MemoryBuffer.h"

#if LLVM_ENABLE_LIBXML2
#include <libxml/xmlreader.h>
#endif

#define TO_XML_CHAR(X) reinterpret_cast<const unsigned char *>(X)
#define FROM_XML_CHAR(X) reinterpret_cast<const char *>(X)

using namespace llvm;
using namespace windows_manifest;

char WindowsManifestError::ID = 0;

WindowsManifestError::WindowsManifestError(const Twine &Msg) : Msg(Msg.str()) {}

void WindowsManifestError::log(raw_ostream &OS) const { OS << Msg; }

class WindowsManifestMerger::WindowsManifestMergerImpl {
public:
  Error merge(MemoryBufferRef Manifest);
  std::unique_ptr<MemoryBuffer> getMergedManifest();

private:
  static void errorCallback(void *Ctx, const char *Format, ...);
  Error getParseError();
#if LLVM_ENABLE_LIBXML2
  struct XmlDeleter {
    void operator()(xmlChar *Ptr) { xmlFree(Ptr); }
    void operator()(xmlDoc *Ptr) { xmlFreeDoc(Ptr); }
  };
  xmlDocPtr CombinedDoc = nullptr;
  std::vector<std::unique_ptr<xmlDoc, XmlDeleter>> MergedDocs;
  bool Merged = false;
  int BufferSize = 0;
  std::unique_ptr<xmlChar, XmlDeleter> Buffer;
#endif
  bool ParseErrorOccurred = false;
};

#if LLVM_ENABLE_LIBXML2

static constexpr std::pair<StringLiteral, StringLiteral> MtNsHrefsPrefixes[] = {
    {"urn:schemas-microsoft-com:asm.v1", "ms_asmv1"},
    {"urn:schemas-microsoft-com:asm.v2", "ms_asmv2"},
    {"urn:schemas-microsoft-com:asm.v3", "ms_asmv3"},
    {"http://schemas.microsoft.com/SMI/2005/WindowsSettings",
     "ms_windowsSettings"},
    {"urn:schemas-microsoft-com:compatibility.v1", "ms_compatibilityv1"}};

static bool xmlStringsEqual(const unsigned char *A, const unsigned char *B) {
  // Handle null pointers.  Comparison of 2 null pointers returns true because
  // this indicates the prefix of a default namespace.
  if (!A || !B)
    return A == B;
  return strcmp(FROM_XML_CHAR(A), FROM_XML_CHAR(B)) == 0;
}

static bool isMergeableElement(const unsigned char *ElementName) {
  for (StringRef S :
       {"application", "assembly", "assemblyIdentity", "compatibility",
        "noInherit", "requestedExecutionLevel", "requestedPrivileges",
        "security", "trustInfo", "windowsSettings"}) {
    if (S == FROM_XML_CHAR(ElementName)) {
      return true;
    }
  }
  return false;
}

// Check if two namespaces refer to the same href.  A node without a namespace
// only matches another node without one.
static bool namespacesMatch(xmlNsPtr Ns1, xmlNsPtr Ns2) {
  if (!Ns1 || !Ns2)
    return Ns1 == Ns2;
  return xmlStringsEqual(Ns1->href, Ns2->href);
}

// Find a child of Parent matching Reference by element name AND namespace.
// Matching by name alone would incorrectly fuse same-named elements living in
// different namespaces (e.g. asm.v1 <application> with asm.v3 <application>),
// producing manifests the SxS loader rejects.
static xmlNodePtr getChildWithName(xmlNodePtr Parent, xmlNodePtr Reference) {
  for (xmlNodePtr Child = Parent->children; Child; Child = Child->next) {
    if (xmlStringsEqual(Child->name, Reference->name) &&
        namespacesMatch(Child->ns, Reference->ns)) {
      return Child;
    }
  }
  return nullptr;
}

static xmlAttrPtr getAttribute(xmlNodePtr Node,
                               const unsigned char *AttributeName) {
  for (xmlAttrPtr Attribute = Node->properties; Attribute != nullptr;
       Attribute = Attribute->next) {
    if (xmlStringsEqual(Attribute->name, AttributeName)) {
      return Attribute;
    }
  }
  return nullptr;
}

// Check if namespace specified by HRef1 overrides that of HRef2.
static bool namespaceOverrides(const unsigned char *HRef1,
                               const unsigned char *HRef2) {
  auto HRef1Position = llvm::find_if(
      MtNsHrefsPrefixes, [=](const std::pair<StringRef, StringRef> &Element) {
        return xmlStringsEqual(HRef1, TO_XML_CHAR(Element.first.data()));
      });
  auto HRef2Position = llvm::find_if(
      MtNsHrefsPrefixes, [=](const std::pair<StringRef, StringRef> &Element) {
        return xmlStringsEqual(HRef2, TO_XML_CHAR(Element.first.data()));
      });
  return HRef1Position < HRef2Position;
}

// Search for prefix-defined namespace specified by HRef, starting on Node and
// continuing recursively upwards.  Returns the namespace or nullptr if not
// found.
static xmlNsPtr search(const unsigned char *HRef, xmlNodePtr Node) {
  for (xmlNsPtr Def = Node->nsDef; Def; Def = Def->next) {
    if (Def->prefix && xmlStringsEqual(Def->href, HRef)) {
      return Def;
    }
  }
  if (Node->parent) {
    return search(HRef, Node->parent);
  }
  return nullptr;
}

// Return the canonical prefix that corresponds to the HRef, or nullptr if
// HRef is not a recognized URI.  (Returning the HRef itself as the prefix,
// as this once did, produces invalid XML of the form xmlns:http://...=...)
static const unsigned char *getPrefixForHref(const unsigned char *HRef) {
  for (auto &Ns : MtNsHrefsPrefixes) {
    if (xmlStringsEqual(HRef, TO_XML_CHAR(Ns.first.data()))) {
      return TO_XML_CHAR(Ns.second.data());
    }
  }
  return nullptr;
}

// Search for prefix-defined namespace specified by HRef, starting on Node and
// continuing recursively upwards.  If it is found, then return it.  If it is
// not found, then prefix-define that namespace on the node and return a
// reference to it.  The prefix is chosen as: the canonical prefix for
// recognized URIs, otherwise the original prefix the namespace had in its
// source document (OldPrefix), otherwise a synthesized unique one.
static Expected<xmlNsPtr>
searchOrDefine(const unsigned char *HRef, xmlNodePtr Node,
               const unsigned char *OldPrefix = nullptr) {
  if (xmlNsPtr Def = search(HRef, Node))
    return Def;
  const unsigned char *Prefix = getPrefixForHref(HRef);
  if (!Prefix)
    Prefix = OldPrefix;
  if (Prefix) {
    if (xmlNsPtr Def = xmlNewNs(Node, HRef, Prefix))
      return Def;
    // The prefix is already bound to a different href on this node; fall
    // through and synthesize a fresh one.
  }
  char Buf[32];
  for (unsigned Suffix = 0; Suffix < 1000; ++Suffix) {
    snprintf(Buf, sizeof(Buf), "ns%u", Suffix);
    if (xmlNsPtr Def = xmlNewNs(Node, HRef, TO_XML_CHAR(Buf)))
      return Def;
  }
  return make_error<WindowsManifestError>("failed to create new namespace");
}

// Set the namespace of OrigionalAttribute on OriginalNode to be that of
// AdditionalAttribute's.
static Error copyAttributeNamespace(xmlAttrPtr OriginalAttribute,
                                    xmlNodePtr OriginalNode,
                                    xmlAttrPtr AdditionalAttribute) {

  Expected<xmlNsPtr> ExplicitOrError =
      searchOrDefine(AdditionalAttribute->ns->href, OriginalNode,
                     AdditionalAttribute->ns->prefix);
  if (!ExplicitOrError)
    return ExplicitOrError.takeError();
  OriginalAttribute->ns = std::move(ExplicitOrError.get());
  return Error::success();
}

// Return the corresponding namespace definition for the prefix, defined on the
// given Node.  Returns nullptr if there is no such definition.
static xmlNsPtr getNamespaceWithPrefix(const unsigned char *Prefix,
                                       xmlNodePtr Node) {
  if (Node == nullptr)
    return nullptr;
  for (xmlNsPtr Def = Node->nsDef; Def; Def = Def->next) {
    if (xmlStringsEqual(Def->prefix, Prefix)) {
      return Def;
    }
  }
  return nullptr;
}

// Search for the closest inheritable default namespace, starting on (and
// including) the Node and traveling upwards through parent nodes.  Returns
// nullptr if there are no inheritable default namespaces.
static xmlNsPtr getClosestDefault(xmlNodePtr Node) {
  if (xmlNsPtr Ret = getNamespaceWithPrefix(nullptr, Node))
    return Ret;
  if (Node->parent == nullptr)
    return nullptr;
  return getClosestDefault(Node->parent);
}

// Merge the attributes of AdditionalNode into OriginalNode.  If attributes
// with identical types are present, they are not duplicated but rather if
// their values are not consistent and error is thrown.  In addition, the
// higher priority namespace is used for each attribute, EXCEPT in the case
// of merging two default namespaces and the lower priority namespace
// definition occurs closer than the higher priority one.
static Error mergeAttributes(xmlNodePtr OriginalNode,
                             xmlNodePtr AdditionalNode) {
  xmlNsPtr ClosestDefault = getClosestDefault(OriginalNode);
  for (xmlAttrPtr Attribute = AdditionalNode->properties; Attribute;
       Attribute = Attribute->next) {
    if (xmlAttrPtr OriginalAttribute =
            getAttribute(OriginalNode, Attribute->name)) {
      if (!xmlStringsEqual(OriginalAttribute->children->content,
                           Attribute->children->content)) {
        return make_error<WindowsManifestError>(
            Twine("conflicting attributes for ") +
            FROM_XML_CHAR(OriginalNode->name));
      }
      if (!Attribute->ns) {
        continue;
      }
      if (!OriginalAttribute->ns) {
        if (auto E = copyAttributeNamespace(OriginalAttribute, OriginalNode,
                                            Attribute)) {
          return E;
        }
        continue;
      }
      if (namespaceOverrides(OriginalAttribute->ns->href,
                             Attribute->ns->href)) {
        // In this case, the original attribute has a higher priority namespace
        // than the incomiing attribute, however the namespace definition of
        // the lower priority namespace occurs first traveling upwards in the
        // tree.  Therefore the lower priority namespace is applied.
        if (!OriginalAttribute->ns->prefix && !Attribute->ns->prefix &&
            ClosestDefault &&
            xmlStringsEqual(Attribute->ns->href, ClosestDefault->href)) {
          if (auto E = copyAttributeNamespace(OriginalAttribute, OriginalNode,
                                              Attribute)) {
            return E;
          }
          continue;
        }
        continue;
        // This covers the case where the incoming attribute has the higher
        // priority.  The higher priority namespace is applied in all cases
        // EXCEPT when both of the namespaces are default inherited, and the
        // closest inherited default is the lower priority one.
      }
      if (Attribute->ns->prefix || OriginalAttribute->ns->prefix ||
          (ClosestDefault && !xmlStringsEqual(OriginalAttribute->ns->href,
                                              ClosestDefault->href))) {
        if (auto E = copyAttributeNamespace(OriginalAttribute, OriginalNode,
                                            Attribute)) {
          return E;
        }
        continue;
      }
      continue;
    }
    // If the incoming attribute is not already found on the node, append it
    // to the end of the properties list.  Also explicitly apply its
    // namespace as a prefix because it might be contained in a separate
    // namespace that doesn't use the attribute.
    xmlAttrPtr NewProp =
        xmlNewProp(OriginalNode, Attribute->name, Attribute->children->content);
    Expected<xmlNsPtr> ExplicitOrError = searchOrDefine(
        Attribute->ns->href, OriginalNode, Attribute->ns->prefix);
    if (!ExplicitOrError)
      return ExplicitOrError.takeError();
    NewProp->ns = std::move(ExplicitOrError.get());
  }
  return Error::success();
}

// Given two nodes, return the one with the higher priority namespace.
static xmlNodePtr getDominantNode(xmlNodePtr Node1, xmlNodePtr Node2) {

  if (!Node1 || !Node1->ns)
    return Node2;
  if (!Node2 || !Node2->ns)
    return Node1;
  if (namespaceOverrides(Node1->ns->href, Node2->ns->href))
    return Node1;
  return Node2;
}

// Checks if this Node's namespace is inherited or one it defined itself.
static bool hasInheritedNs(xmlNodePtr Node) {
  return Node->ns && Node->ns != getNamespaceWithPrefix(Node->ns->prefix, Node);
}

// Check if this Node's namespace is a default namespace that it inherited, as
// opposed to defining itself.
static bool hasInheritedDefaultNs(xmlNodePtr Node) {
  return hasInheritedNs(Node) && Node->ns->prefix == nullptr;
}

// Check if this Node's namespace is a default namespace it defined itself.
static bool hasDefinedDefaultNamespace(xmlNodePtr Node) {
  return Node->ns && (Node->ns == getNamespaceWithPrefix(nullptr, Node));
}

// For the given explicit prefix-definition of a namespace, travel downwards
// from a node recursively, and for every implicit, inherited default usage of
// that namespace replace it with that explicit prefix use.  This is important
// when namespace overriding occurs when merging, so that elements unique to a
// namespace will still stay in that namespace.
static void explicateNamespace(xmlNsPtr PrefixDef, xmlNodePtr Node) {
  // If a node as its own default namespace definition it clearly cannot have
  // inherited the given default namespace, and neither will any of its
  // children.
  if (hasDefinedDefaultNamespace(Node))
    return;
  if (Node->ns && xmlStringsEqual(Node->ns->href, PrefixDef->href) &&
      hasInheritedDefaultNs(Node))
    Node->ns = PrefixDef;
  for (xmlAttrPtr Attribute = Node->properties; Attribute;
       Attribute = Attribute->next) {
    if (Attribute->ns &&
        xmlStringsEqual(Attribute->ns->href, PrefixDef->href)) {
      Attribute->ns = PrefixDef;
    }
  }
  for (xmlNodePtr Child = Node->children; Child; Child = Child->next) {
    explicateNamespace(PrefixDef, Child);
  }
}

// Perform the namespace merge between two nodes.
static Error mergeNamespaces(xmlNodePtr OriginalNode,
                             xmlNodePtr AdditionalNode) {
  // Save the original default namespace definition in case the incoming node
  // overrides it.
  const unsigned char *OriginalDefinedDefaultHref = nullptr;
  if (xmlNsPtr OriginalDefinedDefaultNs =
          getNamespaceWithPrefix(nullptr, OriginalNode)) {
    OriginalDefinedDefaultHref = xmlStrdup(OriginalDefinedDefaultNs->href);
  }
  const unsigned char *NewDefinedDefaultHref = nullptr;
  // Copy all namespace definitions.  There can only be one default namespace
  // definition per node, so the higher priority one takes precedence in the
  // case of collision.
  for (xmlNsPtr Def = AdditionalNode->nsDef; Def; Def = Def->next) {
    if (xmlNsPtr OriginalNsDef =
            getNamespaceWithPrefix(Def->prefix, OriginalNode)) {
      if (!Def->prefix) {
        if (namespaceOverrides(Def->href, OriginalNsDef->href)) {
          NewDefinedDefaultHref = TO_XML_CHAR(strdup(FROM_XML_CHAR(Def->href)));
        }
      } else if (!xmlStringsEqual(OriginalNsDef->href, Def->href)) {
        return make_error<WindowsManifestError>(
            Twine("conflicting namespace definitions for ") +
            FROM_XML_CHAR(Def->prefix));
      }
    } else {
      xmlNsPtr NewDef = xmlCopyNamespace(Def);
      NewDef->next = OriginalNode->nsDef;
      OriginalNode->nsDef = NewDef;
    }
  }

  // Check whether the original node or the incoming node has the higher
  // priority namespace.  Depending on which one is dominant, we will have
  // to recursively apply namespace changes down to children of the original
  // node.
  xmlNodePtr DominantNode = getDominantNode(OriginalNode, AdditionalNode);
  xmlNodePtr NonDominantNode =
      DominantNode == OriginalNode ? AdditionalNode : OriginalNode;
  if (DominantNode == OriginalNode) {
    if (OriginalDefinedDefaultHref) {
      xmlNsPtr NonDominantDefinedDefault =
          getNamespaceWithPrefix(nullptr, NonDominantNode);
      // In this case, both the nodes defined a default namespace.  However
      // the lower priority node ended up having a higher priority default
      // definition.  This can occur if the higher priority node is prefix
      // namespace defined.  In this case we have to define an explicit
      // prefix for the overridden definition and apply it to all children
      // who relied on that definition.
      if (NonDominantDefinedDefault &&
          namespaceOverrides(NonDominantDefinedDefault->href,
                             OriginalDefinedDefaultHref)) {
        Expected<xmlNsPtr> EC =
            searchOrDefine(OriginalDefinedDefaultHref, DominantNode);
        if (!EC) {
          return EC.takeError();
        }
        xmlNsPtr PrefixDominantDefinedDefault = std::move(EC.get());
        explicateNamespace(PrefixDominantDefinedDefault, DominantNode);
      }
      // In this case the node with a higher priority namespace did not have a
      // default namespace definition, but the lower priority node did.  In this
      // case the new default namespace definition is copied.  A side effect of
      // this is that all children will suddenly find themselves in a different
      // default namespace.  To maintain correctness we need to ensure that all
      // children now explicitly refer to the namespace that they had previously
      // implicitly inherited.
    } else if (getNamespaceWithPrefix(nullptr, NonDominantNode)) {
      if (DominantNode->parent) {
        xmlNsPtr ClosestDefault = getClosestDefault(DominantNode->parent);
        Expected<xmlNsPtr> EC =
            searchOrDefine(ClosestDefault->href, DominantNode);
        if (!EC) {
          return EC.takeError();
        }
        xmlNsPtr ExplicitDefault = std::move(EC.get());
        explicateNamespace(ExplicitDefault, DominantNode);
      }
    }
  } else {
    // Covers case where the incoming node has a default namespace definition
    // that overrides the original node's namespace.  This always leads to
    // the original node receiving that new default namespace.
    if (hasDefinedDefaultNamespace(DominantNode)) {
      NonDominantNode->ns = getNamespaceWithPrefix(nullptr, NonDominantNode);
    } else {
      // This covers the case where the incoming node either has a prefix
      // namespace, or an inherited default namespace.  Since the namespace
      // may not yet be defined in the original tree we do a searchOrDefine
      // for it, and then set the namespace equal to it.
      Expected<xmlNsPtr> EC = searchOrDefine(
          DominantNode->ns->href, NonDominantNode, DominantNode->ns->prefix);
      if (!EC) {
        return EC.takeError();
      }
      xmlNsPtr Explicit = std::move(EC.get());
      NonDominantNode->ns = Explicit;
    }
    // This covers cases where the incoming dominant node HAS a default
    // namespace definition, but MIGHT NOT NECESSARILY be in that namespace.
    if (xmlNsPtr DominantDefaultDefined =
            getNamespaceWithPrefix(nullptr, DominantNode)) {
      if (OriginalDefinedDefaultHref) {
        if (namespaceOverrides(DominantDefaultDefined->href,
                               OriginalDefinedDefaultHref)) {
          // In this case, the incoming node's default definition overrides
          // the original default definition, all children who relied on that
          // definition must be updated accordingly.
          Expected<xmlNsPtr> EC =
              searchOrDefine(OriginalDefinedDefaultHref, NonDominantNode);
          if (!EC) {
            return EC.takeError();
          }
          xmlNsPtr ExplicitDefault = std::move(EC.get());
          explicateNamespace(ExplicitDefault, NonDominantNode);
        }
      } else {
        // The original did not define a default definition, however the new
        // default definition still applies to all children, so they must be
        // updated to explicitly refer to the namespace they had previously
        // been inheriting implicitly.
        xmlNsPtr ClosestDefault = getClosestDefault(NonDominantNode);
        Expected<xmlNsPtr> EC =
            searchOrDefine(ClosestDefault->href, NonDominantNode);
        if (!EC) {
          return EC.takeError();
        }
        xmlNsPtr ExplicitDefault = std::move(EC.get());
        explicateNamespace(ExplicitDefault, NonDominantNode);
      }
    }
  }
  if (NewDefinedDefaultHref) {
    xmlNsPtr OriginalNsDef = getNamespaceWithPrefix(nullptr, OriginalNode);
    xmlFree(const_cast<unsigned char *>(OriginalNsDef->href));
    OriginalNsDef->href = NewDefinedDefaultHref;
  }
  xmlFree(const_cast<unsigned char *>(OriginalDefinedDefaultHref));
  return Error::success();
}

static bool isRecognizedNamespace(const unsigned char *NsHref) {
  for (auto &Ns : MtNsHrefsPrefixes) {
    if (xmlStringsEqual(NsHref, TO_XML_CHAR(Ns.first.data()))) {
      return true;
    }
  }
  return false;
}

static bool hasRecognizedNamespace(xmlNodePtr Node) {
  return isRecognizedNamespace(Node->ns->href);
}

// Skip nodes that carry no semantic weight for equality comparison:
// comments and whitespace-only text.
static xmlNodePtr nextRelevantNode(xmlNodePtr Node) {
  while (Node && (Node->type == XML_COMMENT_NODE ||
                  (Node->type == XML_TEXT_NODE && xmlIsBlankNode(Node)))) {
    Node = Node->next;
  }
  return Node;
}

static bool attributesEqual(xmlNodePtr A, xmlNodePtr B) {
  int CountA = 0;
  for (xmlAttrPtr Attr = A->properties; Attr; Attr = Attr->next, ++CountA) {
    xmlAttrPtr Other = getAttribute(B, Attr->name);
    if (!Other || !namespacesMatch(Attr->ns, Other->ns) ||
        !xmlStringsEqual(Attr->children ? Attr->children->content : nullptr,
                         Other->children ? Other->children->content : nullptr))
      return false;
  }
  int CountB = 0;
  for (xmlAttrPtr Attr = B->properties; Attr; Attr = Attr->next)
    ++CountB;
  return CountA == CountB;
}

// Structural equality: same element name and namespace href, equal attribute
// sets, and recursively equal children (ignoring comments and whitespace).
static bool nodesDeepEqual(xmlNodePtr A, xmlNodePtr B) {
  if (A->type != B->type)
    return false;
  if (A->type == XML_TEXT_NODE)
    return xmlStringsEqual(A->content, B->content);
  if (A->type != XML_ELEMENT_NODE)
    return false;
  if (!xmlStringsEqual(A->name, B->name) || !namespacesMatch(A->ns, B->ns) ||
      !attributesEqual(A, B))
    return false;
  xmlNodePtr ChildA = nextRelevantNode(A->children);
  xmlNodePtr ChildB = nextRelevantNode(B->children);
  while (ChildA && ChildB) {
    if (!nodesDeepEqual(ChildA, ChildB))
      return false;
    ChildA = nextRelevantNode(ChildA->next);
    ChildB = nextRelevantNode(ChildB->next);
  }
  return !ChildA && !ChildB;
}

// Check if Parent already contains an element structurally identical to
// Reference.  Used to drop duplicates during merging: two inputs commonly
// supply the same windows setting (e.g. <dpiAware>true</dpiAware> from both a
// user manifest and a build-system-generated fragment), and duplicated
// settings make the SxS loader reject the manifest.  mt.exe deduplicates
// these as well.
static bool hasEqualChild(xmlNodePtr Parent, xmlNodePtr Reference) {
  for (xmlNodePtr Child = Parent->children; Child; Child = Child->next) {
    if (Child->type == XML_ELEMENT_NODE && nodesDeepEqual(Child, Reference)) {
      return true;
    }
  }
  return false;
}

// Ensure a node's inherited namespace is actually defined in the tree it
// resides in.
static Error reconcileNamespaces(xmlNodePtr Node) {
  if (!Node) {
    return Error::success();
  }
  if (hasInheritedNs(Node)) {
    Expected<xmlNsPtr> ExplicitOrError =
        searchOrDefine(Node->ns->href, Node, Node->ns->prefix);
    if (!ExplicitOrError) {
      return ExplicitOrError.takeError();
    }
    xmlNsPtr Explicit = std::move(ExplicitOrError.get());
    Node->ns = Explicit;
  }
  for (xmlNodePtr Child = Node->children; Child; Child = Child->next) {
    if (auto E = reconcileNamespaces(Child)) {
      return E;
    }
  }
  return Error::success();
}

// Recursively merge the two given manifest trees, depending on which elements
// are of a mergeable type, and choose namespaces according to which have
// higher priority.
static Error treeMerge(xmlNodePtr OriginalRoot, xmlNodePtr AdditionalRoot) {
  if (auto E = mergeAttributes(OriginalRoot, AdditionalRoot))
    return E;
  if (auto E = mergeNamespaces(OriginalRoot, AdditionalRoot))
    return E;
  xmlNodePtr AdditionalFirstChild = AdditionalRoot->children;
  xmlNode StoreNext;
  for (xmlNodePtr Child = AdditionalFirstChild; Child; Child = Child->next) {
    xmlNodePtr OriginalChildWithName;
    if (!isMergeableElement(Child->name) ||
        !(OriginalChildWithName = getChildWithName(OriginalRoot, Child)) ||
        !hasRecognizedNamespace(Child)) {
      if (Child->type == XML_ELEMENT_NODE &&
          hasEqualChild(OriginalRoot, Child)) {
        // An identical element is already present; drop the duplicate.
        StoreNext.next = Child->next;
        xmlUnlinkNode(Child);
        xmlFreeNode(Child);
        Child = &StoreNext;
        continue;
      }
      StoreNext.next = Child->next;
      xmlUnlinkNode(Child);
      if (!xmlAddChild(OriginalRoot, Child)) {
        return make_error<WindowsManifestError>(Twine("could not merge ") +
                                                FROM_XML_CHAR(Child->name));
      }
      if (auto E = reconcileNamespaces(Child)) {
        return E;
      }
      Child = &StoreNext;
    } else if (auto E = treeMerge(OriginalChildWithName, Child)) {
      return E;
    }
  }
  return Error::success();
}

static void stripComments(xmlNodePtr Root) {
  xmlNode StoreNext;
  for (xmlNodePtr Child = Root->children; Child; Child = Child->next) {
    if (!xmlStringsEqual(Child->name, TO_XML_CHAR("comment"))) {
      stripComments(Child);
      continue;
    }
    StoreNext.next = Child->next;
    xmlNodePtr Remove = Child;
    Child = &StoreNext;
    xmlUnlinkNode(Remove);
    xmlFreeNode(Remove);
  }
}

// libxml2 assumes that attributes do not inherit default namespaces, whereas
// the original  mt.exe does make this assumption.  This function reconciles
// this by setting all attributes to have the inherited default namespace.
static void setAttributeNamespaces(xmlNodePtr Node) {
  for (xmlAttrPtr Attribute = Node->properties; Attribute;
       Attribute = Attribute->next) {
    if (!Attribute->ns) {
      Attribute->ns = getClosestDefault(Node);
    }
  }
  for (xmlNodePtr Child = Node->children; Child; Child = Child->next) {
    setAttributeNamespaces(Child);
  }
}

// The merging process may create too many prefix defined namespaces.  This
// function removes all unnecessary ones from the tree.
static void checkAndStripPrefixes(xmlNodePtr Node,
                                  std::vector<xmlNsPtr> &RequiredPrefixes) {
  for (xmlNodePtr Child = Node->children; Child; Child = Child->next) {
    checkAndStripPrefixes(Child, RequiredPrefixes);
  }
  if (Node->ns && Node->ns->prefix != nullptr) {
    xmlNsPtr ClosestDefault = getClosestDefault(Node);
    if (ClosestDefault &&
        xmlStringsEqual(ClosestDefault->href, Node->ns->href)) {
      Node->ns = ClosestDefault;
    } else if (!llvm::is_contained(RequiredPrefixes, Node->ns)) {
      RequiredPrefixes.push_back(Node->ns);
    }
  }
  for (xmlAttrPtr Attribute = Node->properties; Attribute;
       Attribute = Attribute->next) {
    if (Attribute->ns && Attribute->ns->prefix != nullptr) {
      xmlNsPtr ClosestDefault = getClosestDefault(Node);
      if (ClosestDefault &&
          xmlStringsEqual(ClosestDefault->href, Attribute->ns->href)) {
        Attribute->ns = ClosestDefault;
      } else if (!llvm::is_contained(RequiredPrefixes, Attribute->ns)) {
        RequiredPrefixes.push_back(Attribute->ns);
      }
    }
  }
  xmlNsPtr Prev;
  xmlNs Temp;
  for (xmlNsPtr Def = Node->nsDef; Def; Def = Def->next) {
    if (!Def->prefix || llvm::is_contained(RequiredPrefixes, Def)) {
      Prev = Def;
      continue;
    }
    if (Def == Node->nsDef) {
      Node->nsDef = Def->next;
    } else {
      Prev->next = Def->next;
    }
    Temp.next = Def->next;
    xmlFreeNs(Def);
    Def = &Temp;
  }
}

Error WindowsManifestMerger::WindowsManifestMergerImpl::merge(
    MemoryBufferRef Manifest) {
  if (Merged)
    return make_error<WindowsManifestError>(
        "merge after getMergedManifest is not supported");
  if (Manifest.getBufferSize() == 0)
    return make_error<WindowsManifestError>(
        "attempted to merge empty manifest");
  xmlSetGenericErrorFunc((void *)this,
                         WindowsManifestMergerImpl::errorCallback);
  std::unique_ptr<xmlDoc, XmlDeleter> ManifestXML(xmlReadMemory(
      Manifest.getBufferStart(), Manifest.getBufferSize(), "manifest.xml",
      nullptr, XML_PARSE_NOBLANKS | XML_PARSE_NODICT));
  xmlSetGenericErrorFunc(nullptr, nullptr);
  if (auto E = getParseError())
    return E;
  xmlNodePtr AdditionalRoot = xmlDocGetRootElement(ManifestXML.get());
  stripComments(AdditionalRoot);
  setAttributeNamespaces(AdditionalRoot);
  if (CombinedDoc == nullptr) {
    CombinedDoc = ManifestXML.get();
  } else {
    xmlNodePtr CombinedRoot = xmlDocGetRootElement(CombinedDoc);
    if (!xmlStringsEqual(CombinedRoot->name, AdditionalRoot->name) ||
        !isMergeableElement(AdditionalRoot->name) ||
        !hasRecognizedNamespace(AdditionalRoot)) {
      return make_error<WindowsManifestError>("multiple root nodes");
    }
    if (auto E = treeMerge(CombinedRoot, AdditionalRoot)) {
      return E;
    }
  }
  MergedDocs.push_back(std::move(ManifestXML));
  return Error::success();
}

std::unique_ptr<MemoryBuffer>
WindowsManifestMerger::WindowsManifestMergerImpl::getMergedManifest() {
  if (!Merged) {
    Merged = true;

    if (!CombinedDoc)
      return nullptr;

    xmlNodePtr CombinedRoot = xmlDocGetRootElement(CombinedDoc);
    std::vector<xmlNsPtr> RequiredPrefixes;
    checkAndStripPrefixes(CombinedRoot, RequiredPrefixes);
    std::unique_ptr<xmlDoc, XmlDeleter> OutputDoc(
        xmlNewDoc((const unsigned char *)"1.0"));
    xmlDocSetRootElement(OutputDoc.get(), CombinedRoot);
    assert(nullptr == xmlDocGetRootElement(CombinedDoc));

    xmlChar *Buff = nullptr;
    xmlDocDumpFormatMemoryEnc(OutputDoc.get(), &Buff, &BufferSize, "UTF-8", 1);
    Buffer.reset(Buff);
  }

  return BufferSize ? MemoryBuffer::getMemBufferCopy(StringRef(
                          FROM_XML_CHAR(Buffer.get()), (size_t)BufferSize))
                    : nullptr;
}

bool windows_manifest::isAvailable() { return true; }

#else

Error WindowsManifestMerger::WindowsManifestMergerImpl::merge(
    MemoryBufferRef Manifest) {
  return make_error<WindowsManifestError>("no libxml2");
}

std::unique_ptr<MemoryBuffer>
WindowsManifestMerger::WindowsManifestMergerImpl::getMergedManifest() {
  return nullptr;
}

bool windows_manifest::isAvailable() { return false; }

#endif

WindowsManifestMerger::WindowsManifestMerger()
    : Impl(std::make_unique<WindowsManifestMergerImpl>()) {}

WindowsManifestMerger::~WindowsManifestMerger() = default;

Error WindowsManifestMerger::merge(MemoryBufferRef Manifest) {
  return Impl->merge(Manifest);
}

std::unique_ptr<MemoryBuffer> WindowsManifestMerger::getMergedManifest() {
  return Impl->getMergedManifest();
}

void WindowsManifestMerger::WindowsManifestMergerImpl::errorCallback(
    void *Ctx, const char *Format, ...) {
  auto *Merger = (WindowsManifestMergerImpl *)Ctx;
  Merger->ParseErrorOccurred = true;
}

Error WindowsManifestMerger::WindowsManifestMergerImpl::getParseError() {
  if (!ParseErrorOccurred)
    return Error::success();
  return make_error<WindowsManifestError>("invalid xml document");
}

//===-- WindowsManifestMerger.cpp ------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// This file implements the .manifest merger class.
//
//===---------------------------------------------------------------------===//

#include "llvm/WindowsManifest/WindowsManifestMerger.h"
#include "llvm/Support/MemoryBuffer.h"

#include <stdarg.h>

#define TO_XML_CHAR(X) reinterpret_cast<const unsigned char *>(X)
#define FROM_XML_CHAR(X) reinterpret_cast<const char *>(X)

using namespace llvm;

namespace llvm {

char WindowsManifestError::ID = 0;

WindowsManifestError::WindowsManifestError(const Twine &Msg) : Msg(Msg.str()) {}

void WindowsManifestError::log(raw_ostream &OS) const { OS << Msg; }

#if LLVM_LIBXML2_ENABLED
static bool xmlStringsEqual(const unsigned char *A, const unsigned char *B) {
  return strcmp(FROM_XML_CHAR(A), FROM_XML_CHAR(B)) == 0;
}
#endif

bool isMergeableElement(const unsigned char *ElementName) {
  for (StringRef S : {"application", "assembly", "assemblyIdentity",
                      "compatibility", "noInherit", "requestedExecutionLevel",
                      "requestedPrivileges", "security", "trustInfo"}) {
    if (S == FROM_XML_CHAR(ElementName))
      return true;
  }
  return false;
}

XMLNodeImpl getChildWithName(XMLNodeImpl Parent,
                             const unsigned char *ElementName) {
#if LLVM_LIBXML2_ENABLED
  for (XMLNodeImpl Child = Parent->children; Child; Child = Child->next)
    if (xmlStringsEqual(Child->name, ElementName)) {
      return Child;
    }
#endif
  return nullptr;
}

const unsigned char *getAttribute(XMLNodeImpl Node,
                                  const unsigned char *AttributeName) {
#if LLVM_LIBXML2_ENABLED
  for (xmlAttrPtr Attribute = Node->properties; Attribute != nullptr;
       Attribute = Attribute->next) {
    if (xmlStringsEqual(Attribute->name, AttributeName))
      return Attribute->children->content;
  }
#endif
  return nullptr;
}

Error mergeAttributes(XMLNodeImpl OriginalNode, XMLNodeImpl AdditionalNode) {
#if LLVM_LIBXML2_ENABLED
  for (xmlAttrPtr Attribute = AdditionalNode->properties; Attribute != nullptr;
       Attribute = Attribute->next) {
    if (const unsigned char *OriginalValue =
            getAttribute(OriginalNode, Attribute->name)) {
      // Attributes of the same name must also have the same value.  Otherwise
      // an error is thrown.
      if (!xmlStringsEqual(OriginalValue, Attribute->children->content))
        return make_error<WindowsManifestError>(
            Twine("conflicting attributes for ") +
            FROM_XML_CHAR(OriginalNode->name));
    } else {
      char *NameCopy = strdup(FROM_XML_CHAR(Attribute->name));
      char *ContentCopy = strdup(FROM_XML_CHAR(Attribute->children->content));
      xmlNewProp(OriginalNode, TO_XML_CHAR(NameCopy), TO_XML_CHAR(ContentCopy));
    }
  }
#endif
  return Error::success();
}

Error treeMerge(XMLNodeImpl OriginalRoot, XMLNodeImpl AdditionalRoot) {
#if LLVM_LIBXML2_ENABLED
  XMLNodeImpl AdditionalFirstChild = AdditionalRoot->children;
  xmlNode StoreNext;
  for (XMLNodeImpl Child = AdditionalFirstChild; Child; Child = Child->next) {
    XMLNodeImpl OriginalChildWithName;
    if (!isMergeableElement(Child->name) ||
        !(OriginalChildWithName =
              getChildWithName(OriginalRoot, Child->name))) {
      StoreNext.next = Child->next;
      xmlUnlinkNode(Child);
      if (!xmlAddChild(OriginalRoot, Child))
        return make_error<WindowsManifestError>(Twine("could not merge ") +
                                                FROM_XML_CHAR(Child->name));
      Child = &StoreNext;
    } else if (auto E = treeMerge(OriginalChildWithName, Child)) {
      return E;
    }
  }
  if (auto E = mergeAttributes(OriginalRoot, AdditionalRoot))
    return E;
#endif
  return Error::success();
}

void stripCommentsAndText(XMLNodeImpl Root) {
#if LLVM_LIBXML2_ENABLED
  xmlNode StoreNext;
  for (XMLNodeImpl Child = Root->children; Child; Child = Child->next) {
    if (!xmlStringsEqual(Child->name, TO_XML_CHAR("text")) &&
        !xmlStringsEqual(Child->name, TO_XML_CHAR("comment"))) {
      stripCommentsAndText(Child);
    } else {
      StoreNext.next = Child->next;
      XMLNodeImpl Remove = Child;
      Child = &StoreNext;
      xmlUnlinkNode(Remove);
      xmlFreeNode(Remove);
    }
  }
#endif
}

WindowsManifestMerger::~WindowsManifestMerger() {
#if LLVM_LIBXML2_ENABLED
  for (auto &Doc : MergedDocs)
    xmlFreeDoc(Doc);
#endif
}

Error WindowsManifestMerger::merge(const MemoryBuffer &Manifest) {
#if LLVM_LIBXML2_ENABLED
  if (Manifest.getBufferSize() == 0)
    return make_error<WindowsManifestError>(
        "attempted to merge empty manifest");
  xmlSetGenericErrorFunc((void *)this, WindowsManifestMerger::errorCallback);
  XMLDocumentImpl ManifestXML =
      xmlReadMemory(Manifest.getBufferStart(), Manifest.getBufferSize(),
                    "manifest.xml", nullptr, XML_PARSE_NOBLANKS);
  xmlSetGenericErrorFunc(nullptr, nullptr);
  if (auto E = getParseError())
    return E;
  XMLNodeImpl AdditionalRoot = xmlDocGetRootElement(ManifestXML);
  stripCommentsAndText(AdditionalRoot);
  if (CombinedDoc == nullptr) {
    CombinedDoc = ManifestXML;
  } else {
    XMLNodeImpl CombinedRoot = xmlDocGetRootElement(CombinedDoc);
    if (xmlStringsEqual(CombinedRoot->name, AdditionalRoot->name) &&
        isMergeableElement(AdditionalRoot->name)) {
      if (auto E = treeMerge(CombinedRoot, AdditionalRoot)) {
        return E;
      }
    } else {
      return make_error<WindowsManifestError>("multiple root nodes");
    }
  }
  MergedDocs.push_back(ManifestXML);
#endif
  return Error::success();
}

std::unique_ptr<MemoryBuffer> WindowsManifestMerger::getMergedManifest() {
#if LLVM_LIBXML2_ENABLED
  unsigned char *XmlBuff;
  int BufferSize = 0;
  if (CombinedDoc) {
    std::unique_ptr<xmlDoc> OutputDoc(xmlNewDoc((const unsigned char *)"1.0"));
    xmlDocSetRootElement(OutputDoc.get(), xmlDocGetRootElement(CombinedDoc));
    xmlKeepBlanksDefault(0);
    xmlDocDumpFormatMemory(OutputDoc.get(), &XmlBuff, &BufferSize, 1);
  }
  if (BufferSize == 0)
    return nullptr;
  return MemoryBuffer::getMemBuffer(
      StringRef(FROM_XML_CHAR(XmlBuff), (size_t)BufferSize));
#else
  return nullptr;
#endif
}

void WindowsManifestMerger::errorCallback(void *Ctx, const char *Format, ...) {
  auto *Merger = (WindowsManifestMerger *)Ctx;
  Merger->ParseErrorOccurred = true;
}

Error WindowsManifestMerger::getParseError() {
  if (!ParseErrorOccurred)
    return Error::success();
  return make_error<WindowsManifestError>("invalid xml document");
}

} // namespace llvm

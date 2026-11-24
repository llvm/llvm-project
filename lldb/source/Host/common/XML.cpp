//===-- XML.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Host/XML.h"
#include "lldb/Host/XMLParser.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace lldb;
using namespace lldb_private;

struct lldb_private::XMLNodeImpl {
  bool is_element = false;
  /// The element name, empty for text nodes.
  std::string name;
  /// The character data of a text node.
  std::string text;
  std::vector<std::pair<std::string, std::string>> attributes;
  XMLNodeImpl *parent = nullptr;
  XMLNodeImpl *next = nullptr;
  std::vector<std::unique_ptr<XMLNodeImpl>> children;

  ~XMLNodeImpl() {
    std::vector<std::unique_ptr<XMLNodeImpl>> pending = std::move(children);
    while (!pending.empty()) {
      std::vector<std::unique_ptr<XMLNodeImpl>> grandchildren =
          std::move(pending.back()->children);
      pending.pop_back();
      for (std::unique_ptr<XMLNodeImpl> &node : grandchildren)
        pending.push_back(std::move(node));
    }
  }
};

namespace {
class TreeBuilder : public xml::Handler {
public:
  explicit TreeBuilder(XMLNodeImpl &document) : m_current(&document) {}

  void StartElement(llvm::StringRef name,
                    llvm::ArrayRef<xml::Attribute> attributes) override {
    XMLNodeImpl &node = AddChild();
    node.is_element = true;
    node.name = name.str();
    for (const xml::Attribute &attr : attributes)
      node.attributes.emplace_back(attr.name.str(), attr.value.str());
    m_current = &node;
  }

  void EndElement(llvm::StringRef name) override {
    m_current = m_current->parent;
  }

  void Characters(llvm::StringRef text) override {
    if (m_current->children.empty() || m_current->children.back()->is_element)
      AddChild();
    m_current->children.back()->text.append(text);
  }

private:
  XMLNodeImpl &AddChild() {
    auto node = std::make_unique<XMLNodeImpl>();
    node->parent = m_current;
    if (!m_current->children.empty())
      m_current->children.back()->next = node.get();
    m_current->children.push_back(std::move(node));
    return *m_current->children.back();
  }

  XMLNodeImpl *m_current;
};
} // namespace

#pragma mark-- XMLDocument

XMLDocument::XMLDocument() = default;

XMLDocument::~XMLDocument() = default;

void XMLDocument::Clear() { m_document.reset(); }

bool XMLDocument::IsValid() const { return m_document != nullptr; }

bool XMLDocument::ParseFile(const char *path) {
  Clear();
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    m_errors.Printf("%s: %s\n", path, buffer.getError().message().c_str());
    return false;
  }
  return ParseMemory((*buffer)->getBufferStart(), (*buffer)->getBufferSize(),
                     path);
}

bool XMLDocument::ParseMemory(const char *xml, size_t xml_length,
                              const char *url) {
  Clear();
  auto document = std::make_unique<XMLNodeImpl>();
  TreeBuilder builder(*document);
  if (llvm::Error error =
          xml::Parse(llvm::StringRef(xml, xml_length), builder)) {
    m_errors.Printf("%s: %s\n", url, llvm::toString(std::move(error)).c_str());
    return false;
  }
  m_document = std::move(document);
  return true;
}

XMLNode XMLDocument::GetRootElement(const char *required_name) {
  if (IsValid()) {
    XMLNode root_node(m_document->children.front().get());
    if (required_name) {
      llvm::StringRef actual_name = root_node.GetName();
      if (actual_name == required_name)
        return root_node;
    } else {
      return root_node;
    }
  }
  return XMLNode();
}

llvm::StringRef XMLDocument::GetErrors() const { return m_errors.GetString(); }

bool XMLDocument::XMLEnabled() { return true; }

#pragma mark-- XMLNode

XMLNode::XMLNode() = default;

XMLNode::XMLNode(const XMLNodeImpl *node) : m_node(node) {}

XMLNode::~XMLNode() = default;

void XMLNode::Clear() { m_node = nullptr; }

XMLNode XMLNode::GetParent() const {
  if (IsValid())
    return XMLNode(m_node->parent);
  else
    return XMLNode();
}

XMLNode XMLNode::GetSibling() const {
  if (IsValid())
    return XMLNode(m_node->next);
  else
    return XMLNode();
}

XMLNode XMLNode::GetChild() const {
  if (IsValid() && !m_node->children.empty())
    return XMLNode(m_node->children.front().get());
  else
    return XMLNode();
}

std::string XMLNode::GetAttributeValue(const char *name,
                                       const char *fail_value) const {
  std::string attr_value;
  if (IsValid()) {
    for (const auto &[attr_name, value] : m_node->attributes)
      if (attr_name == name)
        attr_value = value;
  } else {
    if (fail_value)
      attr_value = fail_value;
  }
  return attr_value;
}

bool XMLNode::GetAttributeValueAsUnsigned(const char *name, uint64_t &value,
                                          uint64_t fail_value, int base) const {
  value = fail_value;
  return llvm::to_integer(GetAttributeValue(name, ""), value, base);
}

void XMLNode::ForEachChildNode(NodeCallback const &callback) const {
  if (IsValid())
    GetChild().ForEachSiblingNode(callback);
}

void XMLNode::ForEachChildElement(NodeCallback const &callback) const {
  XMLNode child = GetChild();
  if (child)
    child.ForEachSiblingElement(callback);
}

void XMLNode::ForEachChildElementWithName(const char *name,
                                          NodeCallback const &callback) const {
  XMLNode child = GetChild();
  if (child)
    child.ForEachSiblingElementWithName(name, callback);
}

void XMLNode::ForEachAttribute(AttributeCallback const &callback) const {
  if (IsValid()) {
    for (const auto &[name, value] : m_node->attributes)
      if (!callback(name, value))
        return;
  }
}

void XMLNode::ForEachSiblingNode(NodeCallback const &callback) const {
  if (IsValid()) {
    // iterate through all siblings
    for (const XMLNodeImpl *node = m_node; node; node = node->next) {
      if (!callback(XMLNode(node)))
        return;
    }
  }
}

void XMLNode::ForEachSiblingElement(NodeCallback const &callback) const {
  if (IsValid()) {
    // iterate through all siblings
    for (const XMLNodeImpl *node = m_node; node; node = node->next) {
      // we are looking for element nodes only
      if (!node->is_element)
        continue;

      if (!callback(XMLNode(node)))
        return;
    }
  }
}

void XMLNode::ForEachSiblingElementWithName(
    const char *name, NodeCallback const &callback) const {
  if (IsValid()) {
    // iterate through all siblings
    for (const XMLNodeImpl *node = m_node; node; node = node->next) {
      // we are looking for element nodes only
      if (!node->is_element)
        continue;

      // If name is nullptr, we take all nodes of type "t", else just the ones
      // whose name matches. Elements always have a name.
      if (!name || node->name != name)
        continue;

      if (!callback(XMLNode(node)))
        return;
    }
  }
}

llvm::StringRef XMLNode::GetName() const {
  if (IsValid())
    return m_node->name;
  return llvm::StringRef();
}

bool XMLNode::GetElementText(std::string &text) const {
  text.clear();
  if (IsValid()) {
    bool success = false;
    if (m_node->is_element) {
      // check child is a text node
      for (const auto &node : m_node->children) {
        if (!node->is_element) {
          text.append(node->text);
          success = true;
        }
      }
    }
    return success;
  }
  return false;
}

bool XMLNode::GetElementTextAsUnsigned(uint64_t &value, uint64_t fail_value,
                                       int base) const {
  std::string text;

  value = fail_value;
  return GetElementText(text) && llvm::to_integer(text, value, base);
}

bool XMLNode::GetElementTextAsFloat(double &value, double fail_value) const {
  std::string text;

  value = fail_value;
  return GetElementText(text) && llvm::to_float(text, value);
}

bool XMLNode::NameIs(const char *name) const {
  return IsValid() && name && m_node->name == name;
}

XMLNode XMLNode::FindFirstChildElementWithName(const char *name) const {
  XMLNode result_node;

  ForEachChildElementWithName(name,
                              [&result_node](const XMLNode &node) -> bool {
                                result_node = node;
                                // Stop iterating, we found the node we wanted
                                return false;
                              });

  return result_node;
}

bool XMLNode::IsValid() const { return m_node != nullptr; }

bool XMLNode::IsElement() const { return IsValid() && m_node->is_element; }

XMLNode XMLNode::GetElementForPath(const NamePath &path) {
  if (IsValid()) {
    if (path.empty())
      return *this;
    else {
      XMLNode node = FindFirstChildElementWithName(path[0].c_str());
      const size_t n = path.size();
      for (size_t i = 1; node && i < n; ++i)
        node = node.FindFirstChildElementWithName(path[i].c_str());
      return node;
    }
  }

  return XMLNode();
}

#pragma mark-- ApplePropertyList

ApplePropertyList::ApplePropertyList() : m_xml_doc(), m_dict_node() {}

ApplePropertyList::ApplePropertyList(const char *path)
    : m_xml_doc(), m_dict_node() {
  ParseFile(path);
}

ApplePropertyList::~ApplePropertyList() = default;

llvm::StringRef ApplePropertyList::GetErrors() const {
  return m_xml_doc.GetErrors();
}

bool ApplePropertyList::ParseFile(const char *path) {
  if (m_xml_doc.ParseFile(path)) {
    XMLNode plist = m_xml_doc.GetRootElement("plist");
    if (plist) {
      plist.ForEachChildElementWithName("dict",
                                        [this](const XMLNode &dict) -> bool {
                                          this->m_dict_node = dict;
                                          return false; // Stop iterating
                                        });
      return (bool)m_dict_node;
    }
  }
  return false;
}

bool ApplePropertyList::IsValid() const { return (bool)m_dict_node; }

bool ApplePropertyList::GetValueAsString(const char *key,
                                         std::string &value) const {
  XMLNode value_node = GetValueNode(key);
  if (value_node)
    return ApplePropertyList::ExtractStringFromValueNode(value_node, value);
  return false;
}

XMLNode ApplePropertyList::GetValueNode(const char *key) const {
  XMLNode value_node;

  if (IsValid()) {
    m_dict_node.ForEachChildElementWithName(
        "key", [key, &value_node](const XMLNode &key_node) -> bool {
          std::string key_name;
          if (key_node.GetElementText(key_name)) {
            if (key_name == key) {
              value_node = key_node.GetSibling();
              while (value_node && !value_node.IsElement())
                value_node = value_node.GetSibling();
              return false; // Stop iterating
            }
          }
          return true; // Keep iterating
        });
  }
  return value_node;
}

bool ApplePropertyList::ExtractStringFromValueNode(const XMLNode &node,
                                                   std::string &value) {
  value.clear();
  if (node.IsValid()) {
    llvm::StringRef element_name = node.GetName();
    if (element_name == "true" || element_name == "false") {
      // The text value _is_ the element name itself...
      value = element_name.str();
      return true;
    } else if (element_name == "dict" || element_name == "array")
      return false; // dictionaries and arrays have no text value, so we fail
    else
      return node.GetElementText(value);
  }
  return false;
}

static StructuredData::ObjectSP CreatePlistValue(XMLNode node) {
  llvm::StringRef element_name = node.GetName();
  if (element_name == "array") {
    std::shared_ptr<StructuredData::Array> array_sp(
        new StructuredData::Array());
    node.ForEachChildElement([&array_sp](const XMLNode &node) -> bool {
      array_sp->AddItem(CreatePlistValue(node));
      return true; // Keep iterating through all child elements of the array
    });
    return array_sp;
  } else if (element_name == "dict") {
    XMLNode key_node;
    std::shared_ptr<StructuredData::Dictionary> dict_sp(
        new StructuredData::Dictionary());
    node.ForEachChildElement(
        [&key_node, &dict_sp](const XMLNode &node) -> bool {
          if (node.NameIs("key")) {
            // This is a "key" element node
            key_node = node;
          } else {
            // This is a value node
            if (key_node) {
              std::string key_name;
              key_node.GetElementText(key_name);
              dict_sp->AddItem(key_name, CreatePlistValue(node));
              key_node.Clear();
            }
          }
          return true; // Keep iterating through all child elements of the
                       // dictionary
        });
    return dict_sp;
  } else if (element_name == "real") {
    double value = 0.0;
    node.GetElementTextAsFloat(value);
    return StructuredData::ObjectSP(new StructuredData::Float(value));
  } else if (element_name == "integer") {
    uint64_t value = 0;
    node.GetElementTextAsUnsigned(value, 0, 0);
    return StructuredData::ObjectSP(new StructuredData::UnsignedInteger(value));
  } else if ((element_name == "string") || (element_name == "data") ||
             (element_name == "date")) {
    std::string text;
    node.GetElementText(text);
    return StructuredData::ObjectSP(
        new StructuredData::String(std::move(text)));
  } else if (element_name == "true") {
    return StructuredData::ObjectSP(new StructuredData::Boolean(true));
  } else if (element_name == "false") {
    return StructuredData::ObjectSP(new StructuredData::Boolean(false));
  }
  return StructuredData::ObjectSP(new StructuredData::Null());
}

StructuredData::ObjectSP ApplePropertyList::GetStructuredData() {
  StructuredData::ObjectSP root_sp;
  if (IsValid()) {
    return CreatePlistValue(m_dict_node);
  }
  return root_sp;
}

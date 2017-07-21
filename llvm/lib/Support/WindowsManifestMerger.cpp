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

#include "llvm/Support/WindowsManifestMerger.h"
#include "llvm/Support/MemoryBuffer.h"

#include <stdarg.h>

namespace llvm {

char WindowsManifestError::ID = 0;

WindowsManifestError::WindowsManifestError(const Twine &Msg) : Msg(Msg.str()) {}

void WindowsManifestError::log(raw_ostream &OS) const { OS << Msg; }

Error WindowsManifestMerger::merge(const MemoryBuffer &Manifest) {
#if LLVM_LIBXML2_ENABLED
  xmlSetGenericErrorFunc((void *)this, WindowsManifestMerger::errorCallback);
  XMLDocumentImpl ManifestXML =
      xmlReadMemory(Manifest.getBufferStart(), Manifest.getBufferSize(),
                    "manifest.xml", nullptr, 0);
  xmlSetGenericErrorFunc(nullptr, nullptr);
  if (auto E = getParseError())
    return E;
  CombinedRoot = xmlDocGetRootElement(ManifestXML);
#endif
  return Error::success();
}

std::unique_ptr<MemoryBuffer> WindowsManifestMerger::getMergedManifest() {
#if LLVM_LIBXML2_ENABLED
  unsigned char *XmlBuff;
  int BufferSize = 0;
  if (CombinedRoot) {
    std::unique_ptr<xmlDoc> OutputDoc(xmlNewDoc((const unsigned char *)"1.0"));
    xmlDocSetRootElement(OutputDoc.get(), CombinedRoot);
    xmlDocDumpMemory(OutputDoc.get(), &XmlBuff, &BufferSize);
  }
  if (BufferSize == 0)
    return nullptr;
  return MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(XmlBuff), (size_t)BufferSize));
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

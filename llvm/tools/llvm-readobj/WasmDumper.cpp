//===-- WasmDumper.cpp - Wasm-specific object file dumper -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Wasm-specific dumper for llvm-readobj.
//
//===----------------------------------------------------------------------===//

#include "Error.h"
#include "ObjDumper.h"
#include "llvm/Object/Wasm.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace object;

namespace {

const char *wasmSectionTypeToString(uint32_t Type) {
#define ECase(X)                                                               \
  case wasm::WASM_SEC_##X:                                                     \
    return #X;
  switch (Type) {
    ECase(CUSTOM);
    ECase(TYPE);
    ECase(IMPORT);
    ECase(FUNCTION);
    ECase(TABLE);
    ECase(MEMORY);
    ECase(GLOBAL);
    ECase(EXPORT);
    ECase(START);
    ECase(ELEM);
    ECase(CODE);
    ECase(DATA);
  }
#undef ECase
  return "";
}

class WasmDumper : public ObjDumper {
public:
  WasmDumper(const WasmObjectFile *Obj, ScopedPrinter &Writer)
      : ObjDumper(Writer), Obj(Obj) {}

  void printFileHeaders() override {
    W.printHex("Version", Obj->getHeader().Version);
  }

  void printSections() override {
    ListScope Group(W, "Sections");
    for (const SectionRef &Section : Obj->sections()) {
      const WasmSection &WasmSec = Obj->getWasmSection(Section);
      DictScope SectionD(W, "Section");
      const char *Type = wasmSectionTypeToString(WasmSec.Type);
      W.printHex("Type", Type, WasmSec.Type);
      W.printNumber("Size", (uint64_t)WasmSec.Content.size());
      W.printNumber("Offset", WasmSec.Offset);
      if (WasmSec.Type == wasm::WASM_SEC_CUSTOM) {
        W.printString("Name", WasmSec.Name);
      }
    }
  }
  void printRelocations() override { llvm_unreachable("unimplemented"); }
  void printSymbols() override { llvm_unreachable("unimplemented"); }
  void printDynamicSymbols() override { llvm_unreachable("unimplemented"); }
  void printUnwindInfo() override { llvm_unreachable("unimplemented"); }
  void printStackMap() const override { llvm_unreachable("unimplemented"); }

private:
  const WasmObjectFile *Obj;
};
}

namespace llvm {

std::error_code createWasmDumper(const object::ObjectFile *Obj,
                                 ScopedPrinter &Writer,
                                 std::unique_ptr<ObjDumper> &Result) {
  const WasmObjectFile *WasmObj = dyn_cast<WasmObjectFile>(Obj);
  assert(WasmObj && "createWasmDumper called with non-wasm object");

  Result.reset(new WasmDumper(WasmObj, Writer));
  return readobj_error::success;
}

} // namespace llvm

#include "ObjDumper.h"
#include "llvm-readobj.h"
#include "llvm/Object/GOFFObjectFile.h"
#include "llvm/Support/ScopedPrinter.h"

using namespace llvm;
using namespace llvm::object;

namespace {

class GOFFDumper : public ObjDumper {
public:
  GOFFDumper(const GOFFObjectFile *Obj, ScopedPrinter &Writer)
      : ObjDumper(Writer, Obj->getFileName()), Obj(Obj) {}

  void printFileHeaders() override {}
  void printSectionHeaders() override;
  void printRelocations() override;
  void printSymbols(bool ExtraSymInfo) override;
  void printDynamicSymbols() override;
  void printUnwindInfo() override {}
  void printStackMap() const override {}

  const object::GOFFObjectFile *getGOFFObject() const { return Obj; };

private:
  void printSymbol(const SymbolRef &Sym);

  const GOFFObjectFile *Obj;
};

} // End anonymous namespace

namespace llvm {
std::unique_ptr<ObjDumper> createGOFFDumper(const object::GOFFObjectFile &Obj,
                                            ScopedPrinter &Writer) {
  return std::make_unique<GOFFDumper>(&Obj, Writer);
}

} // namespace llvm

void GOFFDumper::printSymbol(const SymbolRef &Symbol) {
  DictScope D(W, "Symbol");

  Expected<StringRef> SymbolNameOrErr = Obj->getSymbolName(Symbol);
  if (!SymbolNameOrErr)
    reportError(SymbolNameOrErr.takeError(), Obj->getFileName());
  W.printString("Name", SymbolNameOrErr.get());
  Expected<uint64_t> SymVal = Symbol.getValue();
  if (!SymVal)
    reportError(SymVal.takeError(), Obj->getFileName());
  W.printNumber("Value", *SymVal);
  W.printNumber("Alignment", Symbol.getAlignment());
}

void GOFFDumper::printSectionHeaders() {
  ListScope SectionsD(W, "Sections");
  for (const SectionRef &Sec : Obj->sections()) {
    StringRef Name = unwrapOrError(Obj->getFileName(), Sec.getName());

    DictScope D(W, "Section");
    W.printNumber("Index", Sec.getIndex());
    W.printString("Name", Name);
    W.printHex("Address", Sec.getAddress());
    W.printHex("Size", Sec.getSize());
    W.printNumber("Alignment", Sec.getAlignment().value());

    if (opts::SectionSymbols) {
      ListScope D(W, "Symbols");
      for (const SymbolRef &Symbol : Obj->symbols()) {
        if (!Sec.containsSymbol(Symbol))
          continue;

        printSymbol(Symbol);
      }
    }

    if (opts::SectionData) {
      StringRef Data = unwrapOrError(Obj->getFileName(), Sec.getContents());
      W.printBinaryBlock("SectionData", Data);
    }
  }
}

void GOFFDumper::printSymbols(bool) {
  ListScope Group(W, "Symbols");

  for (const SymbolRef &Symbol : Obj->symbols())
    printSymbol(Symbol);
}

void GOFFDumper::printDynamicSymbols() { ListScope Group(W, "DynamicSymbols"); }

void GOFFDumper::printRelocations() {}

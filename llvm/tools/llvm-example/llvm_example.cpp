#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace object;

cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input file>"), cl::Required);

Error findFunctions(ObjectFile *Obj) {
    for (const SectionRef &Section : Obj->sections()) {
        StringRef Contents = cantFail(Section.getContents());
        StringRef SectionName = cantFail(Section.getName());

        // Skip non-executable sections
        outs() << SectionName <<"\n";
        if (SectionName != ".text")
            continue;

        Expected<StringRef> SectionDataOrErr = Section.getContents();
        if (!SectionDataOrErr)
            return SectionDataOrErr.takeError();
        StringRef SectionData = SectionDataOrErr.get();

        const char *Data = SectionData.data();
        
        const char *End = Data + SectionData.size();
        // Iterate through the section's data looking for potential function entry points
        for (const char *IData = Data; IData < End; ++IData) {
            // Check for common function prologue patterns
            if (IData[0] == '\x55' && IData[1] == '\x48' && IData[2] == '\x89' && IData[3] == '\xe5') {
                outs() << "Potential Function Entry Point: 0x" << Twine::utohexstr((IData - SectionData.data())+Section.getAddress()) << "\n";
            }
        }
    }
    return Error::success();
}

int main(int argc, char **argv) {
    cl::ParseCommandLineOptions(argc, argv, "Function Finder\n");

    // Open the object file
    Expected<OwningBinary<Binary>> BinaryOrErr = createBinary(InputFilename);
    if (!BinaryOrErr) {
        logAllUnhandledErrors(BinaryOrErr.takeError(), outs(), "Function Finder");
        return 1;
    }
    Binary &Bin = *BinaryOrErr.get().getBinary();

    // Attempt to parse the object file
    if (ObjectFile *Obj = dyn_cast<ObjectFile>(&Bin)) {
        if (Error Err = findFunctions(Obj)) {
            logAllUnhandledErrors(std::move(Err), outs(), "Function Finder");
            return 1;
        }
    } else {
        errs() << "Unsupported file format\n";
        return 1;
    }

    return 0;
}
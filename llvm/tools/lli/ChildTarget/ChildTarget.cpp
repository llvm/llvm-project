#include "llvm/ExecutionEngine/Orc/OrcArchitectureSupport.h"
#include "llvm/ExecutionEngine/Orc/OrcRemoteTargetServer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Process.h"
#include <sstream>

#include "../RemoteJITUtils.h"

using namespace llvm;
using namespace llvm::orc;
using namespace llvm::sys;

#ifdef __x86_64__
typedef OrcX86_64 HostOrcArch;
#else
typedef OrcGenericArchitecture HostOrcArch;
#endif

int main(int argc, char *argv[]) {

  if (argc != 3) {
    errs() << "Usage: " << argv[0] << " <input fd> <output fd>\n";
    return 1;
  }

  int InFD;
  int OutFD;
  {
    std::istringstream InFDStream(argv[1]), OutFDStream(argv[2]);
    InFDStream >> InFD;
    OutFDStream >> OutFD;
  }

  if (sys::DynamicLibrary::LoadLibraryPermanently(nullptr)) {
    errs() << "Error loading program symbols.\n";
    return 1;
  }

  auto SymbolLookup = [](const std::string &Name) {
    return RTDyldMemoryManager::getSymbolAddressInProcess(Name);
  };

  FDRPCChannel Channel(InFD, OutFD);
  typedef remote::OrcRemoteTargetServer<FDRPCChannel, HostOrcArch> JITServer;
  JITServer Server(Channel, SymbolLookup);

  while (1) {
    JITServer::JITProcId Id = JITServer::InvalidId;
    if (auto EC = Server.getNextProcId(Id)) {
      errs() << "Error: " << EC.message() << "\n";
      return 1;
    }
    switch (Id) {
    case JITServer::TerminateSessionId:
      return 0;
    default:
      if (auto EC = Server.handleKnownProcedure(Id)) {
        errs() << "Error: " << EC.message() << "\n";
        return 1;
      }
    }
  }

  close(InFD);
  close(OutFD);

  return 0;
}

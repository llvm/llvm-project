
#include "llvm/ExecutionEngine/Orc/TargetProcess/ExecutorResolver.h"

#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"

namespace llvm::orc {

void DylibSymbolResolver::resolveAsync(
    const RemoteSymbolLookupSet &L,
    ExecutorResolver::YieldResolveResultFn &&OnResolve) {
  std::vector<ExecutorSymbolDef> Result;
  auto DL = sys::DynamicLibrary(Handle.toPtr<void *>());

  for (const auto &E : L) {
    if (E.Name.empty()) {
      if (E.Required)
        OnResolve(
            make_error<StringError>("Required address for empty symbol \"\"",
                                    inconvertibleErrorCode()));
      else
        Result.emplace_back();
    } else {

      const char *DemangledSymName = E.Name.c_str();
#ifdef __APPLE__
      if (E.Name.front() != '_')
        OnResolve(make_error<StringError>(Twine("MachO symbol \"") + E.Name +
                                              "\" missing leading '_'",
                                          inconvertibleErrorCode()));
      ++DemangledSymName;
#endif

      void *Addr = DL.getAddressOfSymbol(DemangledSymName);
      if (!Addr && E.Required)
        OnResolve(make_error<StringError>(Twine("Missing definition for ") +
                                              DemangledSymName,
                                          inconvertibleErrorCode()));

      // FIXME: determine accurate JITSymbolFlags.
      Result.push_back({ExecutorAddr::fromPtr(Addr), JITSymbolFlags::Exported});
    }
  }

  OnResolve(std::move(Result));
}

} // end namespace llvm::orc
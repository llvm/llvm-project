//===-- llvm-config.cpp - LLVM project configuration utility --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This tool encapsulates information about an LLVM project configuration for
// use by other project's build environments (to determine installed path,
// available features, required libraries, etc.).
//
// Note that although this tool *may* be used by some parts of LLVM's build
// itself (i.e., the Makefiles use it to compute required libraries when linking
// tools), this tool is primarily designed to support external projects.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Config/config.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <set>
#include <vector>
#include <unordered_set>

using namespace llvm;

// Include the build time variables we can report to the user. This is generated
// at build time from the BuildVariables.inc.in file by the build system.
#include "BuildVariables.inc"

// Include the component table. This creates an array of struct
// AvailableComponent entries, which record the component name, library name,
// and required components for all of the available libraries.
//
// Not all components define a library, we also use "library groups" as a way to
// create entries for pseudo groups like x86 or all-targets.
#include "LibraryDependencies.inc"

/// \brief Traverse a single component adding to the topological ordering in
/// \arg RequiredLibs.
///
/// \param Name - The component to traverse.
/// \param ComponentMap - A prebuilt map of component names to descriptors.
/// \param VisitedComponents [in] [out] - The set of already visited components.
/// \param RequiredLibs [out] - The ordered list of required
/// libraries.
/// \param GetComponentNames - Get the component names instead of the
/// library name.
static void VisitComponent(const std::string& Name,
                           const StringMap<AvailableComponent*> &ComponentMap,
                           std::set<AvailableComponent*> &VisitedComponents,
                           std::vector<std::string> &RequiredLibs,
                           bool IncludeNonInstalled, bool GetComponentNames,
                           const std::string *ActiveLibDir, bool *HasMissing) {
  // Lookup the component.
  AvailableComponent *AC = ComponentMap.lookup(Name);
  assert(AC && "Invalid component name!");

  // Add to the visited table.
  if (!VisitedComponents.insert(AC).second) {
    // We are done if the component has already been visited.
    return;
  }

  // Only include non-installed components if requested.
  if (!AC->IsInstalled && !IncludeNonInstalled)
    return;

  // Otherwise, visit all the dependencies.
  for (unsigned i = 0; AC->RequiredLibraries[i]; ++i) {
    VisitComponent(AC->RequiredLibraries[i], ComponentMap, VisitedComponents,
                   RequiredLibs, IncludeNonInstalled, GetComponentNames,
                   ActiveLibDir, HasMissing);
  }

  if (GetComponentNames) {
    RequiredLibs.push_back(Name);
    return;
  }

  // Add to the required library list.
  if (AC->Library) {
    if (!IncludeNonInstalled && HasMissing && !*HasMissing && ActiveLibDir) {
      *HasMissing = !sys::fs::exists(*ActiveLibDir + "/" + AC->Library);
    }
    RequiredLibs.push_back(AC->Library);
  }
}

/// \brief Compute the list of required libraries for a given list of
/// components, in an order suitable for passing to a linker (that is, libraries
/// appear prior to their dependencies).
///
/// \param Components - The names of the components to find libraries for.
/// \param IncludeNonInstalled - Whether non-installed components should be
/// reported.
/// \param GetComponentNames - True if one would prefer the component names.
static std::vector<std::string>
ComputeLibsForComponents(const std::vector<StringRef> &Components,
                         bool IncludeNonInstalled, bool GetComponentNames,
                         const std::string *ActiveLibDir, bool *HasMissing) {
  std::vector<std::string> RequiredLibs;
  std::set<AvailableComponent *> VisitedComponents;

  // Build a map of component names to information.
  StringMap<AvailableComponent*> ComponentMap;
  for (unsigned i = 0; i != array_lengthof(AvailableComponents); ++i) {
    AvailableComponent *AC = &AvailableComponents[i];
    ComponentMap[AC->Name] = AC;
  }

  // Visit the components.
  for (unsigned i = 0, e = Components.size(); i != e; ++i) {
    // Users are allowed to provide mixed case component names.
    std::string ComponentLower = Components[i].lower();

    // Validate that the user supplied a valid component name.
    if (!ComponentMap.count(ComponentLower)) {
      llvm::errs() << "llvm-config: unknown component name: " << Components[i]
                   << "\n";
      exit(1);
    }

    VisitComponent(ComponentLower, ComponentMap, VisitedComponents,
                   RequiredLibs, IncludeNonInstalled, GetComponentNames,
                   ActiveLibDir, HasMissing);
  }

  // The list is now ordered with leafs first, we want the libraries to printed
  // in the reverse order of dependency.
  std::reverse(RequiredLibs.begin(), RequiredLibs.end());

  return RequiredLibs;
}

/* *** */

static void usage() {
  errs() << "\
usage: llvm-config <OPTION>... [<COMPONENT>...]\n\
\n\
Get various configuration information needed to compile programs which use\n\
LLVM.  Typically called from 'configure' scripts.  Examples:\n\
  llvm-config --cxxflags\n\
  llvm-config --ldflags\n\
  llvm-config --libs engine bcreader scalaropts\n\
\n\
Options:\n\
  --version         Print LLVM version.\n\
  --prefix          Print the installation prefix.\n\
  --src-root        Print the source root LLVM was built from.\n\
  --obj-root        Print the object root used to build LLVM.\n\
  --bindir          Directory containing LLVM executables.\n\
  --includedir      Directory containing LLVM headers.\n\
  --libdir          Directory containing LLVM libraries.\n\
  --cppflags        C preprocessor flags for files that include LLVM headers.\n\
  --cflags          C compiler flags for files that include LLVM headers.\n\
  --cxxflags        C++ compiler flags for files that include LLVM headers.\n\
  --ldflags         Print Linker flags.\n\
  --system-libs     System Libraries needed to link against LLVM components.\n\
  --libs            Libraries needed to link against LLVM components.\n\
  --libnames        Bare library names for in-tree builds.\n\
  --libfiles        Fully qualified library filenames for makefile depends.\n\
  --components      List of all possible components.\n\
  --targets-built   List of all targets currently built.\n\
  --host-target     Target triple used to configure LLVM.\n\
  --build-mode      Print build mode of LLVM tree (e.g. Debug or Release).\n\
  --assertion-mode  Print assertion mode of LLVM tree (ON or OFF).\n\
  --build-system    Print the build system used to build LLVM (autoconf or cmake).\n\
  --has-rtti        Print whether or not LLVM was built with rtti (YES or NO).\n\
  --shared-mode     Print how the provided components can be collectively linked (`shared` or `static`).\n\
Typical components:\n\
  all               All LLVM libraries (default).\n\
  engine            Either a native JIT or a bitcode interpreter.\n";
  exit(1);
}

/// \brief Compute the path to the main executable.
std::string GetExecutablePath(const char *Argv0) {
  // This just needs to be some symbol in the binary; C++ doesn't
  // allow taking the address of ::main however.
  void *P = (void*) (intptr_t) GetExecutablePath;
  return llvm::sys::fs::getMainExecutable(Argv0, P);
}

/// \brief Expand the semi-colon delimited LLVM_DYLIB_COMPONENTS into
/// the full list of components.
std::vector<std::string> GetAllDyLibComponents(const bool IsInDevelopmentTree,
                                               const bool GetComponentNames) {
  std::vector<StringRef> DyLibComponents;

  StringRef DyLibComponentsStr(LLVM_DYLIB_COMPONENTS);
  size_t Offset = 0;
  while (true) {
    const size_t NextOffset = DyLibComponentsStr.find(';', Offset);
    DyLibComponents.push_back(DyLibComponentsStr.substr(Offset, NextOffset));
    if (NextOffset == std::string::npos) {
      break;
    }
    Offset = NextOffset + 1;
  }

  assert(!DyLibComponents.empty());

  return ComputeLibsForComponents(DyLibComponents,
                                  /*IncludeNonInstalled=*/IsInDevelopmentTree,
                                  GetComponentNames, nullptr, nullptr);
}

int main(int argc, char **argv) {
  std::vector<StringRef> Components;
  bool PrintLibs = false, PrintLibNames = false, PrintLibFiles = false;
  bool PrintSystemLibs = false, PrintSharedMode = false;
  bool HasAnyOption = false;

  // llvm-config is designed to support being run both from a development tree
  // and from an installed path. We try and auto-detect which case we are in so
  // that we can report the correct information when run from a development
  // tree.
  bool IsInDevelopmentTree;
  enum { MakefileStyle, CMakeStyle, CMakeBuildModeStyle } DevelopmentTreeLayout;
  llvm::SmallString<256> CurrentPath(GetExecutablePath(argv[0]));
  std::string CurrentExecPrefix;
  std::string ActiveObjRoot;

  // If CMAKE_CFG_INTDIR is given, honor it as build mode.
  char const *build_mode = LLVM_BUILDMODE;
#if defined(CMAKE_CFG_INTDIR)
  if (!(CMAKE_CFG_INTDIR[0] == '.' && CMAKE_CFG_INTDIR[1] == '\0'))
    build_mode = CMAKE_CFG_INTDIR;
#endif

  // Create an absolute path, and pop up one directory (we expect to be inside a
  // bin dir).
  sys::fs::make_absolute(CurrentPath);
  CurrentExecPrefix = sys::path::parent_path(
    sys::path::parent_path(CurrentPath)).str();

  // Check to see if we are inside a development tree by comparing to possible
  // locations (prefix style or CMake style).
  if (sys::fs::equivalent(CurrentExecPrefix,
                          Twine(LLVM_OBJ_ROOT) + "/" + build_mode)) {
    IsInDevelopmentTree = true;
    DevelopmentTreeLayout = MakefileStyle;

    // If we are in a development tree, then check if we are in a BuildTools
    // directory. This indicates we are built for the build triple, but we
    // always want to provide information for the host triple.
    if (sys::path::filename(LLVM_OBJ_ROOT) == "BuildTools") {
      ActiveObjRoot = sys::path::parent_path(LLVM_OBJ_ROOT);
    } else {
      ActiveObjRoot = LLVM_OBJ_ROOT;
    }
  } else if (sys::fs::equivalent(CurrentExecPrefix, LLVM_OBJ_ROOT)) {
    IsInDevelopmentTree = true;
    DevelopmentTreeLayout = CMakeStyle;
    ActiveObjRoot = LLVM_OBJ_ROOT;
  } else if (sys::fs::equivalent(CurrentExecPrefix,
                                 Twine(LLVM_OBJ_ROOT) + "/bin")) {
    IsInDevelopmentTree = true;
    DevelopmentTreeLayout = CMakeBuildModeStyle;
    ActiveObjRoot = LLVM_OBJ_ROOT;
  } else {
    IsInDevelopmentTree = false;
    DevelopmentTreeLayout = MakefileStyle; // Initialized to avoid warnings.
  }

  // Compute various directory locations based on the derived location
  // information.
  std::string ActivePrefix, ActiveBinDir, ActiveIncludeDir, ActiveLibDir;
  std::string ActiveIncludeOption;
  if (IsInDevelopmentTree) {
    ActiveIncludeDir = std::string(LLVM_SRC_ROOT) + "/include";
    ActivePrefix = CurrentExecPrefix;

    // CMake organizes the products differently than a normal prefix style
    // layout.
    switch (DevelopmentTreeLayout) {
    case MakefileStyle:
      ActivePrefix = ActiveObjRoot;
      ActiveBinDir = ActiveObjRoot + "/" + build_mode + "/bin";
      ActiveLibDir =
          ActiveObjRoot + "/" + build_mode + "/lib" + LLVM_LIBDIR_SUFFIX;
      break;
    case CMakeStyle:
      ActiveBinDir = ActiveObjRoot + "/bin";
      ActiveLibDir = ActiveObjRoot + "/lib" + LLVM_LIBDIR_SUFFIX;
      break;
    case CMakeBuildModeStyle:
      ActivePrefix = ActiveObjRoot;
      ActiveBinDir = ActiveObjRoot + "/bin/" + build_mode;
      ActiveLibDir =
          ActiveObjRoot + "/lib" + LLVM_LIBDIR_SUFFIX + "/" + build_mode;
      break;
    }

    // We need to include files from both the source and object trees.
    ActiveIncludeOption = ("-I" + ActiveIncludeDir + " " +
                           "-I" + ActiveObjRoot + "/include");
  } else {
    ActivePrefix = CurrentExecPrefix;
    ActiveIncludeDir = ActivePrefix + "/include";
    ActiveBinDir = ActivePrefix + "/bin";
    ActiveLibDir = ActivePrefix + "/lib" + LLVM_LIBDIR_SUFFIX;
    ActiveIncludeOption = "-I" + ActiveIncludeDir;
  }

  /// We only use `shared library` mode in cases where the static library form
  /// of the components provided are not available; note however that this is
  /// skipped if we're run from within the build dir. However, once installed,
  /// we still need to provide correct output when the static archives are
  /// removed or, as in the case of CMake's `BUILD_SHARED_LIBS`, never present
  /// in the first place. This can't be done at configure/build time.

  StringRef SharedExt, SharedVersionedExt, SharedDir, SharedPrefix, StaticExt,
    StaticPrefix, StaticDir = "lib";
  const Triple HostTriple(Triple::normalize(LLVM_DEFAULT_TARGET_TRIPLE));
  if (HostTriple.isOSWindows()) {
    SharedExt = "dll";
    SharedVersionedExt = PACKAGE_VERSION ".dll";
    StaticExt = "a";
    SharedDir = ActiveBinDir;
    StaticDir = ActiveLibDir;
    StaticPrefix = SharedPrefix = "lib";
  } else if (HostTriple.isOSDarwin()) {
    SharedExt = "dylib";
    SharedVersionedExt = PACKAGE_VERSION ".dylib";
    StaticExt = "a";
    StaticDir = SharedDir = ActiveLibDir;
    StaticPrefix = SharedPrefix = "lib";
  } else {
    // default to the unix values:
    SharedExt = "so";
    SharedVersionedExt = PACKAGE_VERSION ".so";
    StaticExt = "a";
    StaticDir = SharedDir = ActiveLibDir;
    StaticPrefix = SharedPrefix = "lib";
  }

  const bool BuiltDyLib = (std::strcmp(LLVM_ENABLE_DYLIB, "ON") == 0);

  enum { CMake, AutoConf } ConfigTool;
  if (std::strcmp(LLVM_BUILD_SYSTEM, "cmake") == 0) {
    ConfigTool = CMake;
  } else {
    ConfigTool = AutoConf;
  }

  /// CMake style shared libs, ie each component is in a shared library.
  const bool BuiltSharedLibs =
      (ConfigTool == CMake && std::strcmp(LLVM_ENABLE_SHARED, "ON") == 0);

  bool DyLibExists = false;
  const std::string DyLibName =
    (SharedPrefix + "LLVM-" + SharedVersionedExt).str();

  if (BuiltDyLib) {
    DyLibExists = sys::fs::exists(SharedDir + "/" + DyLibName);
  }

  /// Get the component's library name without the lib prefix and the
  /// extension. Returns true if Lib is in a recognized format.
  auto GetComponentLibraryNameSlice = [&](const StringRef &Lib,
                                          StringRef &Out) {
    if (Lib.startswith("lib")) {
      unsigned FromEnd;
      if (Lib.endswith(StaticExt)) {
        FromEnd = StaticExt.size() + 1;
      } else if (Lib.endswith(SharedExt)) {
        FromEnd = SharedExt.size() + 1;
      } else {
        FromEnd = 0;
      }

      if (FromEnd != 0) {
        Out = Lib.slice(3, Lib.size() - FromEnd);
        return true;
      }
    }

    return false;
  };
  /// Maps Unixizms to the host platform.
  auto GetComponentLibraryFileName = [&](const StringRef &Lib,
                                         const bool ForceShared) {
    std::string LibFileName = Lib;
    StringRef LibName;
    if (GetComponentLibraryNameSlice(Lib, LibName)) {
      if (BuiltSharedLibs || ForceShared) {
        LibFileName = (SharedPrefix + LibName + "." + SharedExt).str();
      } else {
        // default to static
        LibFileName = (StaticPrefix + LibName + "." + StaticExt).str();
      }
    }

    return LibFileName;
  };
  /// Get the full path for a possibly shared component library.
  auto GetComponentLibraryPath = [&](const StringRef &Name,
                                     const bool ForceShared) {
    auto LibFileName = GetComponentLibraryFileName(Name, ForceShared);
    if (BuiltSharedLibs || ForceShared) {
      return (SharedDir + "/" + LibFileName).str();
    } else {
      return (StaticDir + "/" + LibFileName).str();
    }
  };

  raw_ostream &OS = outs();
  for (int i = 1; i != argc; ++i) {
    StringRef Arg = argv[i];

    if (Arg.startswith("-")) {
      HasAnyOption = true;
      if (Arg == "--version") {
        OS << PACKAGE_VERSION << '\n';
      } else if (Arg == "--prefix") {
        OS << ActivePrefix << '\n';
      } else if (Arg == "--bindir") {
        OS << ActiveBinDir << '\n';
      } else if (Arg == "--includedir") {
        OS << ActiveIncludeDir << '\n';
      } else if (Arg == "--libdir") {
        OS << ActiveLibDir << '\n';
      } else if (Arg == "--cppflags") {
        OS << ActiveIncludeOption << ' ' << LLVM_CPPFLAGS << '\n';
      } else if (Arg == "--cflags") {
        OS << ActiveIncludeOption << ' ' << LLVM_CFLAGS << '\n';
      } else if (Arg == "--cxxflags") {
        OS << ActiveIncludeOption << ' ' << LLVM_CXXFLAGS << '\n';
      } else if (Arg == "--ldflags") {
        OS << "-L" << ActiveLibDir << ' ' << LLVM_LDFLAGS << '\n';
      } else if (Arg == "--system-libs") {
        PrintSystemLibs = true;
      } else if (Arg == "--libs") {
        PrintLibs = true;
      } else if (Arg == "--libnames") {
        PrintLibNames = true;
      } else if (Arg == "--libfiles") {
        PrintLibFiles = true;
      } else if (Arg == "--components") {
        /// If there are missing static archives and a dylib was
        /// built, print LLVM_DYLIB_COMPONENTS instead of everything
        /// in the manifest.
        std::vector<std::string> Components;
        for (unsigned j = 0; j != array_lengthof(AvailableComponents); ++j) {
          // Only include non-installed components when in a development tree.
          if (!AvailableComponents[j].IsInstalled && !IsInDevelopmentTree)
            continue;

          Components.push_back(AvailableComponents[j].Name);
          if (AvailableComponents[j].Library && !IsInDevelopmentTree) {
            if (DyLibExists &&
                !sys::fs::exists(GetComponentLibraryPath(
                    AvailableComponents[j].Library, false))) {
              Components = GetAllDyLibComponents(IsInDevelopmentTree, true);
              std::sort(Components.begin(), Components.end());
              break;
            }
          }
        }

        for (unsigned I = 0; I < Components.size(); ++I) {
          if (I) {
            OS << ' ';
          }

          OS << Components[I];
        }
        OS << '\n';
      } else if (Arg == "--targets-built") {
        OS << LLVM_TARGETS_BUILT << '\n';
      } else if (Arg == "--host-target") {
        OS << Triple::normalize(LLVM_DEFAULT_TARGET_TRIPLE) << '\n';
      } else if (Arg == "--build-mode") {
        OS << build_mode << '\n';
      } else if (Arg == "--assertion-mode") {
#if defined(NDEBUG)
        OS << "OFF\n";
#else
        OS << "ON\n";
#endif
      } else if (Arg == "--build-system") {
        OS << LLVM_BUILD_SYSTEM << '\n';
      } else if (Arg == "--has-rtti") {
        OS << LLVM_HAS_RTTI << '\n';
      } else if (Arg == "--shared-mode") {
        PrintSharedMode = true;
      } else if (Arg == "--obj-root") {
        OS << ActivePrefix << '\n';
      } else if (Arg == "--src-root") {
        OS << LLVM_SRC_ROOT << '\n';
      } else {
        usage();
      }
    } else {
      Components.push_back(Arg);
    }
  }

  if (!HasAnyOption)
    usage();

  if (PrintLibs || PrintLibNames || PrintLibFiles || PrintSystemLibs ||
      PrintSharedMode) {

    if (PrintSharedMode && BuiltSharedLibs) {
      OS << "shared\n";
      return 0;
    }

    // If no components were specified, default to "all".
    if (Components.empty())
      Components.push_back("all");

    // Construct the list of all the required libraries.
    bool HasMissing = false;
    std::vector<std::string> RequiredLibs =
        ComputeLibsForComponents(Components,
                                 /*IncludeNonInstalled=*/IsInDevelopmentTree,
                                 false, &ActiveLibDir, &HasMissing);

    if (PrintSharedMode) {
      std::unordered_set<std::string> FullDyLibComponents;
      std::vector<std::string> DyLibComponents =
          GetAllDyLibComponents(IsInDevelopmentTree, false);

      for (auto &Component : DyLibComponents) {
        FullDyLibComponents.insert(Component);
      }
      DyLibComponents.clear();

      for (auto &Lib : RequiredLibs) {
        if (!FullDyLibComponents.count(Lib)) {
          OS << "static\n";
          return 0;
        }
      }
      FullDyLibComponents.clear();

      if (HasMissing && DyLibExists) {
        OS << "shared\n";
        return 0;
      } else {
        OS << "static\n";
        return 0;
      }
    }

    if (PrintLibs || PrintLibNames || PrintLibFiles) {

      auto PrintForLib = [&](const StringRef &Lib, const bool ForceShared) {
        if (PrintLibNames) {
          OS << GetComponentLibraryFileName(Lib, ForceShared);
        } else if (PrintLibFiles) {
          OS << GetComponentLibraryPath(Lib, ForceShared);
        } else if (PrintLibs) {
          // If this is a typical library name, include it using -l.
          StringRef LibName;
          if (Lib.startswith("lib")) {
            if (GetComponentLibraryNameSlice(Lib, LibName)) {
              OS << "-l" << LibName;
            } else {
              OS << "-l:" << GetComponentLibraryFileName(Lib, ForceShared);
            }
          } else {
            // Otherwise, print the full path.
            OS << GetComponentLibraryPath(Lib, ForceShared);
          }
        }
      };

      if (HasMissing && DyLibExists) {
        PrintForLib(DyLibName, true);
      } else {
        for (unsigned i = 0, e = RequiredLibs.size(); i != e; ++i) {
          auto Lib = RequiredLibs[i];
          if (i)
            OS << ' ';

          PrintForLib(Lib, false);
        }
      }
      OS << '\n';
    }

    // Print SYSTEM_LIBS after --libs.
    // FIXME: Each LLVM component may have its dependent system libs.
    if (PrintSystemLibs)
      OS << LLVM_SYSTEM_LIBS << '\n';
  } else if (!Components.empty()) {
    errs() << "llvm-config: error: components given, but unused\n\n";
    usage();
  }

  return 0;
}

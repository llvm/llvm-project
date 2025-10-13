//===- LibraryResolverTest.cpp - Unit tests for LibraryResolver -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryResolver.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/LibraryScanner.h"
#include "llvm/ObjectYAML/MachOYAML.h"
#include "llvm/ObjectYAML/yaml2obj.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Testing/Support/SupportHelpers.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

#if defined(__APPLE__) || defined(__linux__)
// TODO: Add COFF (Windows) support for these tests.
// this facility also works correctly on Windows (COFF),
// so we should eventually enable and run these tests for that platform as well.
namespace {

#if defined(__APPLE__)
constexpr const char *ext = ".dylib";
#elif defined(_WIN32)
constexpr const char *ext = ".dll";
#else
constexpr const char *ext = ".so";
#endif

bool EnvReady = false;

Triple getTargetTriple() {
  auto JTMB = JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
    consumeError(JTMB.takeError());
    return Triple();
  }
  return JTMB->getTargetTriple();
}

static bool CheckHostSupport() {
  auto Triple = getTargetTriple();
  // TODO: Extend support to COFF (Windows) once test setup and YAML conversion
  // are verified.
  if (!Triple.isOSBinFormatMachO() &&
      !(Triple.isOSBinFormatELF() && Triple.getArch() == Triple::x86_64))
    return false;

  return true;
}

std::string getYamlFilePlatformExt() {
  auto Triple = getTargetTriple();
  if (Triple.isOSBinFormatMachO())
    return "_macho";
  else if (Triple.isOSBinFormatELF())
    return "_linux";

  return "";
}

unsigned getYamlDocNum() {
  // auto Triple = getTargetTriple();
  // if (Triple.isOSBinFormatELF())
  //   return 1;

  return 1;
}

class LibraryTestEnvironment : public ::testing::Environment {
  std::vector<std::string> CreatedDylibsDir;
  std::vector<std::string> CreatedDylibs;
  SmallVector<char, 128> DirPath;

public:
  void SetUp() override {
    if (!CheckHostSupport()) {
      EnvReady = false;
      return;
    }

    StringRef ThisFile = __FILE__;
    SmallVector<char, 128> InputDirPath(ThisFile.begin(), ThisFile.end());
    sys::path::remove_filename(InputDirPath);
    sys::path::append(InputDirPath, "Inputs");
    if (!sys::fs::exists(InputDirPath))
      return;

    SmallString<128> UniqueDir;
    sys::path::append(UniqueDir, InputDirPath);
    std::error_code EC = sys::fs::createUniqueDirectory(UniqueDir, DirPath);

    if (EC)
      return;

    // given yamlPath + DylibPath, validate + convert
    auto processYamlToDylib = [&](const SmallVector<char, 128> &YamlPath,
                                  const SmallVector<char, 128> &DylibPath,
                                  unsigned DocNum) -> bool {
      if (!sys::fs::exists(YamlPath)) {
        errs() << "YAML file missing: "
               << StringRef(YamlPath.data(), YamlPath.size()) << "\n";
        EnvReady = false;
        return false;
      }

      auto BufOrErr = MemoryBuffer::getFile(YamlPath);
      if (!BufOrErr) {
        errs() << "Failed to read "
               << StringRef(YamlPath.data(), YamlPath.size()) << ": "
               << BufOrErr.getError().message() << "\n";
        EnvReady = false;
        return false;
      }

      yaml::Input yin(BufOrErr->get()->getBuffer());
      std::error_code EC;
      raw_fd_ostream outFile(StringRef(DylibPath.data(), DylibPath.size()), EC,
                             sys::fs::OF_None);

      if (EC) {
        errs() << "Failed to open "
               << StringRef(DylibPath.data(), DylibPath.size())
               << " for writing: " << EC.message() << "\n";
        EnvReady = false;
        return false;
      }

      if (!yaml::convertYAML(
              yin, outFile,
              [](const Twine &M) {
                // Handle or ignore errors here
                errs() << "Yaml Error :" << M << "\n";
              },
              DocNum)) {
        errs() << "Failed to convert "
               << StringRef(YamlPath.data(), YamlPath.size()) << " to "
               << StringRef(DylibPath.data(), DylibPath.size()) << "\n";
        EnvReady = false;
        return false;
      }

      CreatedDylibsDir.push_back(std::string(sys::path::parent_path(
          StringRef(DylibPath.data(), DylibPath.size()))));
      CreatedDylibs.push_back(std::string(DylibPath.begin(), DylibPath.end()));
      return true;
    };

    std::vector<const char *> LibDirs = {"Z", "A", "B", "C", "D"};

    unsigned DocNum = getYamlDocNum();
    std::string YamlPltExt = getYamlFilePlatformExt();
    for (const auto &LibdirName : LibDirs) {
      // YAML path
      SmallVector<char, 128> YamlPath(InputDirPath.begin(), InputDirPath.end());
      SmallVector<char, 128> YamlFileName;
      YamlFileName.append(LibdirName, LibdirName + strlen(LibdirName));
      YamlFileName.append(YamlPltExt.begin(), YamlPltExt.end());
      sys::path::append(YamlPath, LibdirName, YamlFileName);
      sys::path::replace_extension(YamlPath, ".yaml");

      // dylib path
      SmallVector<char, 128> DylibPath(DirPath.begin(), DirPath.end());
      SmallVector<char, 128> DylibFileName;
      StringRef prefix("lib");
      DylibFileName.append(prefix.begin(), prefix.end());
      DylibFileName.append(LibdirName, LibdirName + strlen(LibdirName));

      sys::path::append(DylibPath, LibdirName);
      if (!sys::fs::exists(DylibPath)) {
        auto EC = sys::fs::create_directory(DylibPath);
        if (EC)
          return;
      }
      sys::path::append(DylibPath, DylibFileName);
      sys::path::replace_extension(DylibPath, ext);
      if (!processYamlToDylib(YamlPath, DylibPath, DocNum))
        return;
    }

    EnvReady = true;
  }

  void TearDown() override { sys::fs::remove_directories(DirPath); }

  std::string getBaseDir() const {
    return std::string(DirPath.begin(), DirPath.end());
  }

  std::vector<std::string> getDylibPaths() const { return CreatedDylibs; }
};

static LibraryTestEnvironment *GlobalEnv =
    static_cast<LibraryTestEnvironment *>(
        ::testing::AddGlobalTestEnvironment(new LibraryTestEnvironment()));

inline std::string libPath(const std::string &BaseDir,
                           const std::string &name) {
#if defined(__APPLE__)
  return BaseDir + "/" + name + ".dylib";
#elif defined(_WIN32)
  return BaseDir + "/" + name + ".dll";
#else
  return BaseDir + "/" + name + ".so";
#endif
}

inline std::string withext(const std::string &lib) {
  SmallString<128> P(lib);
  sys::path::replace_extension(P, ext);
  return P.str().str();
}

inline std::string platformSymbolName(const std::string &name) {
#if defined(__APPLE__)
  return "_" + name; // macOS prepends underscore
#else
  return name;
#endif
}

struct TestLibrary {
  std::string path;
  std::vector<std::string> symbols;
};

class LibraryResolverIT : public ::testing::Test {
protected:
  std::string BaseDir;
  std::unordered_map<std::string, TestLibrary> libs;
  void SetUp() override {
    if (!EnvReady)
      GTEST_SKIP() << "Skipping test: environment setup failed.";

    ASSERT_NE(GlobalEnv, nullptr);
    BaseDir = GlobalEnv->getBaseDir();
    libs["A"] = {libPath(BaseDir, "A/libA"), {platformSymbolName("sayA")}};
    libs["B"] = {libPath(BaseDir, "B/libB"), {platformSymbolName("sayB")}};
    libs["C"] = {libPath(BaseDir, "C/libC"), {platformSymbolName("sayC")}};
    libs["D"] = {libPath(BaseDir, "D/libD"), {platformSymbolName("sayD")}};
    libs["Z"] = {libPath(BaseDir, "Z/libZ"), {platformSymbolName("sayZ")}};
    for (const auto &P : GlobalEnv->getDylibPaths()) {
      if (!sys::fs::exists(P))
        GTEST_SKIP();
    }
  }

  const std::vector<std::string> &sym(const std::string &key) {
    return libs[key].symbols;
  }
  const std::string &lib(const std::string &key) { return libs[key].path; }
  const std::string libdir(const std::string &key) {
    SmallString<512> P(libs[key].path);
    sys::path::remove_filename(P);
    return P.str().str();
  }
  const std::string libname(const std::string &key) {
    return sys::path::filename(libs[key].path).str();
  }
};

// Helper: allow either "sayA" or "_sayA" depending on how your SymbolEnumerator
// reports.
static bool matchesEitherUnderscore(const std::string &got,
                                    const std::string &bare) {
  return got == bare || got == ("_" + bare);
}

// Helper: normalize path ending check (we only care that it resolved to the
// right dylib)
static bool endsWith(const std::string &s, const std::string &suffix) {
  if (s.size() < suffix.size())
    return false;
  return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

// --- 1) SymbolEnumerator enumerates real exports from libC.dylib ---
TEST_F(LibraryResolverIT, EnumerateSymbolsFromARespectsDefaults) {
  const std::string libC = lib("C");

  SymbolEnumeratorOptions opts = SymbolEnumeratorOptions::defaultOptions();

  std::vector<std::string> seen;
  auto onEach = [&](llvm::StringRef sym) -> EnumerateResult {
    seen.emplace_back(sym.str());
    return EnumerateResult::Continue;
  };

  const bool ok = SymbolEnumerator::enumerateSymbols(libC, onEach, opts);
  ASSERT_TRUE(ok) << "enumerateSymbols failed on " << libC;

  // We expect to see sayA (export) and not an undefined reference to printf.
  bool foundSayA = false;
  for (const auto &s : seen) {
    if (matchesEitherUnderscore(s, "sayA")) {
      foundSayA = true;
      break;
    }
  }
  EXPECT_FALSE(foundSayA) << "Expected exported symbol sayA in libC";
}

TEST_F(LibraryResolverIT, EnumerateSymbols_ExportsOnly_DefaultFlags) {
  const std::string libC = lib("C");
  SymbolEnumeratorOptions opts = SymbolEnumeratorOptions::defaultOptions();

  std::vector<std::string> seen;
  auto onEach = [&](llvm::StringRef sym) -> EnumerateResult {
    seen.emplace_back(sym.str());
    return EnumerateResult::Continue;
  };

  ASSERT_TRUE(SymbolEnumerator::enumerateSymbols(libC, onEach, opts));

  // sayC is exported, others are undefined → only sayC expected
  EXPECT_TRUE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayC");
  }));
  EXPECT_FALSE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayA");
  }));
  EXPECT_FALSE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayB");
  }));
  EXPECT_FALSE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayZ");
  }));
}

TEST_F(LibraryResolverIT, EnumerateSymbols_IncludesUndefineds) {
  const std::string libC = lib("C");

  SymbolEnumeratorOptions opts;
  opts.FilterFlags =
      SymbolEnumeratorOptions::IgnoreWeak |
      SymbolEnumeratorOptions::IgnoreIndirect; // no IgnoreUndefined

  std::vector<std::string> seen;
  auto onEach = [&](llvm::StringRef sym) -> EnumerateResult {
    seen.emplace_back(sym.str());
    return EnumerateResult::Continue;
  };

  ASSERT_TRUE(SymbolEnumerator::enumerateSymbols(libC, onEach, opts));

  // Now we should see both sayC (export) and the undefined refs sayA, sayB,
  // sayZ
  EXPECT_TRUE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayC");
  }));
  EXPECT_TRUE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayA");
  }));
  EXPECT_TRUE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayB");
  }));
  EXPECT_TRUE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayZ");
  }));
}

TEST_F(LibraryResolverIT, EnumerateSymbols_IndirectExportRespected) {
  const std::string libD = lib("D");

  SymbolEnumeratorOptions opts;
  opts.FilterFlags = SymbolEnumeratorOptions::IgnoreWeak; // allow indirects

  std::vector<std::string> seen;
  auto onEach = [&](llvm::StringRef sym) -> EnumerateResult {
    seen.emplace_back(sym.str());
    return EnumerateResult::Continue;
  };

  ASSERT_TRUE(SymbolEnumerator::enumerateSymbols(libD, onEach, opts));

  // sayA is re-exported from A, so should appear unless IgnoreIndirect was set
  EXPECT_TRUE(any_of(seen, [&](const std::string &s) {
    return matchesEitherUnderscore(s, "sayA");
  }));
}

// --- 2) Filters: if we remove IgnoreUndefined, we should also see undefineds
// like printf ---
TEST_F(LibraryResolverIT, EnumerateSymbolsIncludesUndefWhenNotIgnored) {
  const std::string libA = lib("A");

  SymbolEnumeratorOptions opts = SymbolEnumeratorOptions::defaultOptions();
  // Start from defaults but allow undefined
  opts.FilterFlags &= ~SymbolEnumeratorOptions::IgnoreUndefined;

  bool sawPrintf = false;
  auto onEach = [&](llvm::StringRef sym) -> EnumerateResult {
    if (matchesEitherUnderscore(sym.str(), "printf") ||
        matchesEitherUnderscore(sym.str(), "puts"))
      sawPrintf = true;
    return EnumerateResult::Continue;
  };

  ASSERT_TRUE(SymbolEnumerator::enumerateSymbols(libA, onEach, opts));
  EXPECT_TRUE(sawPrintf)
      << "Expected to see undefined symbol printf when not filtered";
}

// --- 3) Full resolution via LibraryResolutionDriver/LibraryResolver ---
TEST_F(LibraryResolverIT, DriverResolvesSymbolsToCorrectLibraries) {
  // Create the resolver from real base paths (our fixtures dir)
  auto setup = LibraryResolver::Setup::create({BaseDir});

  // Full system behavior: no mocks
  auto driver = LibraryResolutionDriver::create(setup);
  ASSERT_NE(driver, nullptr);

  // Tell the driver about the scan path kinds (User/System) as your production
  // code expects.
  driver->addScanPath(libdir("A"), PathType::User);
  driver->addScanPath(libdir("B"), PathType::User);
  driver->addScanPath(libdir("Z"), PathType::User);

  // Symbols to resolve (bare names; class handles underscore differences
  // internally)
  std::vector<std::string> symbols = {platformSymbolName("sayA"),
                                      platformSymbolName("sayB"),
                                      platformSymbolName("sayZ")};

  bool callbackRan = false;
  driver->resolveSymbols(symbols, [&](SymbolQuery &query) {
    callbackRan = true;

    // sayA should resolve to A.dylib
    {
      auto lib = query.getResolvedLib(platformSymbolName("sayA"));
      ASSERT_TRUE(lib.has_value()) << "sayA should be resolved";
      EXPECT_TRUE(endsWith(lib->str(), libname("A")))
          << "sayA resolved to: " << lib->str();
    }

    // sayB should resolve to B.dylib
    {
      auto lib = query.getResolvedLib(platformSymbolName("sayB"));
      ASSERT_TRUE(lib.has_value()) << "sayB should be resolved";
      EXPECT_TRUE(endsWith(lib->str(), libname("B")))
          << "sayB resolved to: " << lib->str();
    }

    // sayZ should resolve to B.dylib
    {
      auto lib = query.getResolvedLib(platformSymbolName("sayZ"));
      ASSERT_TRUE(lib.has_value()) << "sayZ should be resolved";
      EXPECT_TRUE(endsWith(lib->str(), libname("Z")))
          << "sayZ resolved to: " << lib->str();
    }

    EXPECT_TRUE(query.allResolved());
  });

  EXPECT_TRUE(callbackRan);
}

// --- 4) Cross-library reference visibility (C references A) ---
TEST_F(LibraryResolverIT, EnumeratorSeesInterLibraryRelationship) {
  const std::string libC = lib("C");

  SymbolEnumeratorOptions onlyUndef = SymbolEnumeratorOptions::defaultOptions();
  // Show only undefined (drop IgnoreUndefined) to see C's reference to sayA
  onlyUndef.FilterFlags &= ~SymbolEnumeratorOptions::IgnoreUndefined;

  bool sawSayAAsUndef = false;
  auto onEach = [&](llvm::StringRef sym) -> EnumerateResult {
    if (matchesEitherUnderscore(sym.str(), "sayA"))
      sawSayAAsUndef = true;
    return EnumerateResult::Continue;
  };

  ASSERT_TRUE(SymbolEnumerator::enumerateSymbols(libC, onEach, onlyUndef));
  EXPECT_TRUE(sawSayAAsUndef)
      << "libC should have an undefined reference to sayA (defined in libA)";
}

// // // --- 5) Optional: stress SymbolQuery with the real resolve flow
// // // And resolve libC dependency libA, libB, libZ ---
TEST_F(LibraryResolverIT, ResolveManySymbols) {
  auto setup = LibraryResolver::Setup::create({BaseDir});
  auto driver = LibraryResolutionDriver::create(setup);
  ASSERT_NE(driver, nullptr);
  driver->addScanPath(libdir("C"), PathType::User);

  // Many duplicates to provoke concurrent updates inside SymbolQuery
  std::vector<std::string> symbols = {
      platformSymbolName("sayA"), platformSymbolName("sayB"),
      platformSymbolName("sayA"), platformSymbolName("sayB"),
      platformSymbolName("sayZ"), platformSymbolName("sayZ"),
      platformSymbolName("sayZ"), platformSymbolName("sayZ"),
      platformSymbolName("sayA"), platformSymbolName("sayB"),
      platformSymbolName("sayA"), platformSymbolName("sayB")};

  bool callbackRan = false;
  driver->resolveSymbols(symbols, [&](SymbolQuery &query) {
    callbackRan = true;
    EXPECT_TRUE(query.isResolved(platformSymbolName("sayA")));
    EXPECT_TRUE(query.isResolved(platformSymbolName("sayB")));
    EXPECT_TRUE(query.isResolved(platformSymbolName("sayZ")));

    auto a = query.getResolvedLib(platformSymbolName("sayA"));
    auto b = query.getResolvedLib(platformSymbolName("sayB"));
    auto z = query.getResolvedLib(platformSymbolName("sayZ"));
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(z.has_value());
    EXPECT_TRUE(endsWith(a->str(), libname("A")));
    EXPECT_TRUE(endsWith(b->str(), libname("B")));
    EXPECT_TRUE(endsWith(z->str(), libname("Z")));
    EXPECT_TRUE(query.allResolved());
  });

  EXPECT_TRUE(callbackRan);
}

// // // --- 5) Optional: stress SymbolQuery with the real resolve flow
// // // And resolve libD dependency libA ---
TEST_F(LibraryResolverIT, ResolveManySymbols2) {
  auto setup = LibraryResolver::Setup::create({BaseDir});
  auto driver = LibraryResolutionDriver::create(setup);
  ASSERT_NE(driver, nullptr);
  driver->addScanPath(libdir("D"), PathType::User);

  // Many duplicates to provoke concurrent updates inside SymbolQuery
  std::vector<std::string> symbols = {
      platformSymbolName("sayA"), platformSymbolName("sayB"),
      platformSymbolName("sayA"), platformSymbolName("sayB"),
      platformSymbolName("sayZ"), platformSymbolName("sayZ"),
      platformSymbolName("sayZ"), platformSymbolName("sayZ"),
      platformSymbolName("sayD"), platformSymbolName("sayD"),
      platformSymbolName("sayA"), platformSymbolName("sayB"),
      platformSymbolName("sayA"), platformSymbolName("sayB")};

  driver->resolveSymbols(symbols, [&](SymbolQuery &query) {
    EXPECT_TRUE(query.isResolved(platformSymbolName("sayA")));
    EXPECT_TRUE(query.isResolved(platformSymbolName("sayD")));

    auto a = query.getResolvedLib(platformSymbolName("sayA"));
    auto d = query.getResolvedLib(platformSymbolName("sayD"));
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(d.has_value());
    EXPECT_TRUE(endsWith(a->str(), libname("A")));
    EXPECT_TRUE(endsWith(d->str(), libname("D")));
    EXPECT_FALSE(query.allResolved());
  });
}

TEST_F(LibraryResolverIT, ScanSingleUserPath) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);
  LibraryScanHelper scanH({}, cache, presolver);

  scanH.addBasePath(libdir("C"), PathType::User);

  std::error_code ec;
  auto libCPathOpt = presolver->resolve(lib("C"), ec);

  if (!libCPathOpt || ec) {
    FAIL();
  }

  std::string libCPath = *libCPathOpt;

  LibraryManager mgr;
  LibraryScanner scanner(scanH, mgr);

  scanner.scanNext(PathType::User, 0);

  bool found = false;
  mgr.forEachLibrary([&](const LibraryInfo &lib) {
    if (lib.getFullPath() == libCPath) {
      found = true;
    }
    return true;
  });
  EXPECT_TRUE(found) << "Expected to find " << libCPath;
}

TEST_F(LibraryResolverIT, ScanAndCheckDeps) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);
  LibraryScanHelper scanH({}, cache, presolver);

  scanH.addBasePath(libdir("C"), PathType::User);

  LibraryManager mgr;
  LibraryScanner scanner(scanH, mgr);

  scanner.scanNext(PathType::User, 0);

  size_t count = 0;
  mgr.forEachLibrary([&](const LibraryInfo &) {
    count++;
    return true;
  });

  EXPECT_GE(count, 3u) << "Should find at least libA in multiple paths";
}

TEST_F(LibraryResolverIT, ScanEmptyPath) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);
  LibraryScanHelper scanH({}, cache, presolver);

  scanH.addBasePath("/tmp/empty", PathType::User);

  LibraryManager mgr;
  LibraryScanner scanner(scanH, mgr);

  scanner.scanNext(PathType::User, 0);

  size_t count = 0;
  mgr.forEachLibrary([&](const LibraryInfo &) {
    count++;
    return true;
  });
  EXPECT_EQ(count, 0u);
}

TEST_F(LibraryResolverIT, PathResolverResolvesKnownPaths) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  std::error_code ec;
  auto missing = presolver->resolve("temp/foo/bar", ec);
  EXPECT_FALSE(missing.has_value()) << "Unexpectedly resolved a bogus path";
  EXPECT_TRUE(ec) << "Expected error resolving path";

  auto DirPath = presolver->resolve(BaseDir, ec);
  ASSERT_TRUE(DirPath.has_value());
  EXPECT_FALSE(ec) << "Expected no error resolving path";
  EXPECT_EQ(*DirPath, BaseDir);

  auto DylibPath = presolver->resolve(lib("C"), ec);
  ASSERT_TRUE(DylibPath.has_value());
  EXPECT_FALSE(ec) << "Expected no error resolving path";
  EXPECT_EQ(*DylibPath, lib("C"));
}

TEST_F(LibraryResolverIT, PathResolverNormalizesDotAndDotDot) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  std::error_code ec;

  // e.g. BaseDir + "/./C/../C/C.dylib" → BaseDir + "/C.dylib"
  std::string messy = BaseDir + "/C/./../C/./libC" + ext;
  auto resolved = presolver->resolve(messy, ec);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_FALSE(ec);
  EXPECT_EQ(*resolved, lib("C")) << "Expected realpath to collapse . and ..";
}

#if !defined(_WIN32)
TEST_F(LibraryResolverIT, PathResolverFollowsSymlinks) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  std::error_code ec;

  // Create a symlink temp -> BaseDir (only if filesystem allows it)
  std::string linkName = BaseDir + withext("/link_to_C");
  std::string target = lib("C");
  ::symlink(target.c_str(), linkName.c_str());

  auto resolved = presolver->resolve(linkName, ec);
  ASSERT_TRUE(resolved.has_value());
  EXPECT_FALSE(ec);
  EXPECT_EQ(*resolved, target);

  ::unlink(linkName.c_str()); // cleanup
}

TEST_F(LibraryResolverIT, PathResolverCachesResults) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  SmallString<128> tmpDylib;
  sys::fs::createUniqueFile(withext("A-copy"), tmpDylib);
  sys::fs::copy_file(lib("A"), tmpDylib);

  std::error_code ec;

  // First resolve -> should populate cache
  auto first = presolver->resolve(tmpDylib, ec);
  ASSERT_TRUE(first.has_value());

  // Forcefully remove the file from disk
  ::unlink(tmpDylib.c_str());

  // Second resolve -> should still succeed from cache
  auto second = presolver->resolve(tmpDylib, ec);
  EXPECT_TRUE(second.has_value());
  EXPECT_EQ(*second, *first);
}
#endif

TEST_F(LibraryResolverIT, LoaderPathSubstitutionAndResolve) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  DylibSubstitutor substitutor;
  substitutor.configure(libdir("C"));
#if defined(__APPLE__)
  // Substitute @loader_path with BaseDir
  std::string substituted =
      substitutor.substitute(withext("@loader_path/libC"));
#elif defined(__linux__)
  // Substitute $origin with BaseDir
  std::string substituted = substitutor.substitute(withext("$ORIGIN/libC"));
#endif
  ASSERT_FALSE(substituted.empty());
  EXPECT_EQ(substituted, lib("C"));

  // Now try resolving the substituted path
  std::error_code ec;
  auto resolved = presolver->resolve(substituted, ec);
  ASSERT_TRUE(resolved.has_value()) << "Expected to resolve substituted dylib";
  EXPECT_EQ(*resolved, lib("C"));
  EXPECT_FALSE(ec) << "Expected no error resolving substituted dylib";
}

TEST_F(LibraryResolverIT, ResolveFromUsrOrSystemPaths) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  DylibPathValidator validator(*presolver);

  std::vector<std::string> Paths = {"/foo/bar/", "temp/foo",  libdir("C"),
                                    libdir("A"), libdir("B"), libdir("Z")};

  SmallVector<StringRef> P(Paths.begin(), Paths.end());

  DylibResolver Resolver(validator);
  Resolver.configure("", {{P, SearchPathType::UsrOrSys}});

  // Check "C"
  auto valOptC = Resolver.resolve("libC", true);
  EXPECT_TRUE(valOptC.has_value());
  EXPECT_EQ(*valOptC, lib("C"));

  auto valOptCdylib = Resolver.resolve(withext("libC"));
  EXPECT_TRUE(valOptCdylib.has_value());
  EXPECT_EQ(*valOptCdylib, lib("C"));

  // Check "A"
  auto valOptA = Resolver.resolve("libA", true);
  EXPECT_TRUE(valOptA.has_value());
  EXPECT_EQ(*valOptA, lib("A"));

  auto valOptAdylib = Resolver.resolve(withext("libA"));
  EXPECT_TRUE(valOptAdylib.has_value());
  EXPECT_EQ(*valOptAdylib, lib("A"));

  // Check "B"
  auto valOptB = Resolver.resolve("libB", true);
  EXPECT_TRUE(valOptB.has_value());
  EXPECT_EQ(*valOptB, lib("B"));

  auto valOptBdylib = Resolver.resolve(withext("libB"));
  EXPECT_TRUE(valOptBdylib.has_value());
  EXPECT_EQ(*valOptBdylib, lib("B"));

  // Check "Z"
  auto valOptZ = Resolver.resolve("libZ", true);
  EXPECT_TRUE(valOptZ.has_value());
  EXPECT_EQ(*valOptZ, lib("Z"));

  auto valOptZdylib = Resolver.resolve(withext("libZ"));
  EXPECT_TRUE(valOptZdylib.has_value());
  EXPECT_EQ(*valOptZdylib, lib("Z"));
}

#if defined(__APPLE__)
TEST_F(LibraryResolverIT, ResolveViaLoaderPathAndRPathSubstitution) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  DylibPathValidator validator(*presolver);

  std::vector<std::string> Paths = {"@loader_path/../A", "@loader_path/../B",
                                    "@loader_path/../D", "@loader_path/../Z"};

  SmallVector<StringRef> P(Paths.begin(), Paths.end());

  DylibResolver Resolver(validator);

  // Use only RPath config
  Resolver.configure(lib("C"), {{P, SearchPathType::RPath}});

  // --- Check A ---
  auto valOptA = Resolver.resolve("@rpath/libA", true);
  EXPECT_TRUE(valOptA.has_value());
  EXPECT_EQ(*valOptA, lib("A"));

  auto valOptAdylib = Resolver.resolve(withext("@rpath/libA"));
  EXPECT_TRUE(valOptAdylib.has_value());
  EXPECT_EQ(*valOptAdylib, lib("A"));

  // --- Check B ---
  auto valOptB = Resolver.resolve("@rpath/libB", true);
  EXPECT_TRUE(valOptB.has_value());
  EXPECT_EQ(*valOptB, lib("B"));

  auto valOptBdylib = Resolver.resolve(withext("@rpath/libB"));
  EXPECT_TRUE(valOptBdylib.has_value());
  EXPECT_EQ(*valOptBdylib, lib("B"));

  // --- Check Z ---
  auto valOptZ = Resolver.resolve("@rpath/libZ", true);
  EXPECT_TRUE(valOptZ.has_value());
  EXPECT_EQ(*valOptZ, lib("Z"));

  auto valOptZdylib = Resolver.resolve(withext("@rpath/libZ"));
  EXPECT_TRUE(valOptZdylib.has_value());
  EXPECT_EQ(*valOptZdylib, lib("Z"));
}
#endif

#if defined(__linux__)
TEST_F(LibraryResolverIT, ResolveViaOriginAndRPathSubstitution) {
  auto cache = std::make_shared<LibraryPathCache>();
  auto presolver = std::make_shared<PathResolver>(cache);

  DylibPathValidator validator(*presolver);

  // On Linux, $ORIGIN works like @loader_path
  std::vector<std::string> Paths = {"$ORIGIN/../A", "$ORIGIN/../B",
                                    "$ORIGIN/../D", "$ORIGIN/../Z"};

  SmallVector<StringRef> P(Paths.begin(), Paths.end());

  DylibResolver Resolver(validator);

  // Use only RPath config
  Resolver.configure(lib("C"), {{P, SearchPathType::RunPath}});

  // --- Check A ---
  auto valOptA = Resolver.resolve("libA", true);
  EXPECT_TRUE(valOptA.has_value());
  EXPECT_EQ(*valOptA, lib("A"));

  auto valOptASO = Resolver.resolve(withext("libA"));
  EXPECT_TRUE(valOptASO.has_value());
  EXPECT_EQ(*valOptASO, lib("A"));

  // --- Check B ---
  auto valOptB = Resolver.resolve("libB", true);
  EXPECT_TRUE(valOptB.has_value());
  EXPECT_EQ(*valOptB, lib("B"));

  auto valOptBSO = Resolver.resolve(withext("libB"));
  EXPECT_TRUE(valOptBSO.has_value());
  EXPECT_EQ(*valOptBSO, lib("B"));

  // --- Check Z ---
  auto valOptZ = Resolver.resolve("libZ", true);
  EXPECT_TRUE(valOptZ.has_value());
  EXPECT_EQ(*valOptZ, lib("Z"));

  auto valOptZSO = Resolver.resolve(withext("libZ"));
  EXPECT_TRUE(valOptZSO.has_value());
  EXPECT_EQ(*valOptZSO, lib("Z"));
}
#endif
} // namespace
#endif // defined(__APPLE__)

//===- llvm-cas-object-format.cpp - Tool for the CAS object format --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CAS/TreeSchema.h"
#include "llvm/MCCAS/MCCASObjectV1.h"
#include "llvm/RemoteCachingService/RemoteCachingService.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::cas;

cl::opt<std::string> CASPath("cas", cl::desc("Path to CAS on disk."));
cl::list<std::string> InputFiles(cl::Positional, cl::desc("Input object"));
cl::opt<bool> Silent("silent", cl::desc("only print final CAS ID"));
cl::opt<std::string> OutputPrefix("output-prefix",
                                  cl::desc("output object file prefix"),
                                  cl::init("."));

enum InputKind {
  MaterializeObjects,
};

cl::opt<InputKind> InputFileKind(
    cl::desc("choose input kind and action:"),
    cl::values(clEnumValN(MaterializeObjects, "materialize-objects",
                          "materialize objects from CAS tree")),
    cl::init(InputKind::MaterializeObjects));

static Error materializeObjectsFromCASTree(ObjectStore &CAS, ObjectProxy ID);

int main(int argc, char *argv[]) {
  ExitOnError ExitOnErr;
  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  RegisterGRPCCAS Y;
  cl::ParseCommandLineOptions(argc, argv);

  std::shared_ptr<ObjectStore> CAS =
      ExitOnErr(createCASFromIdentifier(CASPath));

  for (StringRef IF : InputFiles) {
    ExitOnError ExitOnErr;
    ExitOnErr.setBanner(("llvm-cas-object-format: " + IF + ": ").str());

    switch (InputFileKind) {

    case MaterializeObjects: {
      auto ID = ExitOnErr(CAS->parseID(IF));
      auto Proxy = ExitOnErr(CAS->getProxy(ID));
      ExitOnErr(materializeObjectsFromCASTree(*CAS, Proxy));
      return 0;
    }
    }
  }
}

static Error materializeObjectsFromCASTree(ObjectStore &CAS, ObjectProxy ID) {
  TreeSchema Schema(CAS);
  return Schema.walkFileTreeRecursively(
      CAS, ID.getRef(),
      [&](const NamedTreeEntry &Entry, std::optional<TreeProxy>) -> Error {
        if (Entry.getKind() != TreeEntry::Regular &&
            Entry.getKind() != TreeEntry::Tree) {
          return createStringError(inconvertibleErrorCode(),
                                   "found non-regular entry: " +
                                       Entry.getName());
        }
        SmallString<256> OutputPath(OutputPrefix);
        StringRef ObjFileName = Entry.getName();
        ObjFileName.consume_back(".casid");
        llvm::sys::path::append(OutputPath, ObjFileName);

        if (Entry.getKind() == TreeEntry::Tree) {
          // Check the path exists, if not, create the directory.
          if (!llvm::sys::fs::exists(OutputPath)) {
            if (auto EC = llvm::sys::fs::create_directory(OutputPath))
              return errorCodeToError(EC);
          }
          return Error::success();
        }
        auto ObjRoot = CAS.getProxy(Entry.getRef());
        if (!ObjRoot)
          return ObjRoot.takeError();

        SmallString<50> ContentsStorage;
        raw_svector_ostream ObjOS(ContentsStorage);
        auto Schema = std::make_unique<llvm::mccasformats::v1::MCSchema>(CAS);
        if (Error E = Schema->serializeObjectFile(*ObjRoot, ObjOS))
          return E;
        std::unique_ptr<llvm::FileOutputBuffer> Output;
        if (Error E = llvm::FileOutputBuffer::create(OutputPath,
                                                     ContentsStorage.size())
                          .moveInto(Output))
          return E;
        llvm::copy(ContentsStorage, Output->getBufferStart());
        return Output->commit();
      });
}

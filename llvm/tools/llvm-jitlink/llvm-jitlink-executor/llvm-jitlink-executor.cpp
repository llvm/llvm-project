//===- llvm-jitlink-executor.cpp - Out-of-proc executor for llvm-jitlink -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Simple out-of-process executor for llvm-jitlink.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/Shared/FDRawByteChannel.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/JITLoaderGDB.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/OrcRPCTPCServer.h"
#include "llvm/ExecutionEngine/Orc/TargetProcess/RegisterEHFrames.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

#ifdef LLVM_ON_UNIX

#include <netinet/in.h>
#include <sys/socket.h>

#endif

using namespace llvm;
using namespace llvm::orc;

ExitOnError ExitOnErr;

LLVM_ATTRIBUTE_USED void linkComponents() {
  errs() << (void *)&llvm_orc_registerEHFrameSectionWrapper
         << (void *)&llvm_orc_deregisterEHFrameSectionWrapper
         << (void *)&llvm_orc_registerJITLoaderGDBWrapper;
}

void printErrorAndExit(Twine ErrMsg) {
  errs() << "error: " << ErrMsg.str() << "\n\n"
         << "Usage:\n"
         << "  llvm-jitlink-executor filedescs=<infd>,<outfd> [args...]\n"
         << "  llvm-jitlink-executor listen=<host>:<port> [args...]\n";
  exit(1);
}

int openListener(std::string Host, int Port) {
#ifndef LLVM_ON_UNIX
  // FIXME: Add TCP support for Windows.
  printErrorAndExit("listen option not supported");
  return 0;
#else
  int SockFD = socket(PF_INET, SOCK_STREAM, 0);
  struct sockaddr_in ServerAddr, ClientAddr;
  socklen_t ClientAddrLen = sizeof(ClientAddr);
  memset(&ServerAddr, 0, sizeof(ServerAddr));
  ServerAddr.sin_family = PF_INET;
  ServerAddr.sin_family = INADDR_ANY;
  ServerAddr.sin_port = htons(Port);

  {
    // lose the "Address already in use" error message
    int Yes = 1;
    if (setsockopt(SockFD, SOL_SOCKET, SO_REUSEADDR, &Yes, sizeof(int)) == -1) {
      errs() << "Error calling setsockopt.\n";
      exit(1);
    }
  }

  if (bind(SockFD, (struct sockaddr *)&ServerAddr, sizeof(ServerAddr)) < 0) {
    errs() << "Error on binding.\n";
    exit(1);
  }

  listen(SockFD, 1);
  return accept(SockFD, (struct sockaddr *)&ClientAddr, &ClientAddrLen);
#endif
}

int main(int argc, char *argv[]) {

  ExitOnErr.setBanner(std::string(argv[0]) + ": ");

  int InFD = 0;
  int OutFD = 0;

  if (argc < 2)
    printErrorAndExit("insufficient arguments");
  else {
    StringRef Arg1 = argv[1];
    StringRef SpecifierType, Specifier;
    std::tie(SpecifierType, Specifier) = Arg1.split('=');
    if (SpecifierType == "filedescs") {
      StringRef FD1Str, FD2Str;
      std::tie(FD1Str, FD2Str) = Specifier.split(',');
      if (FD1Str.getAsInteger(10, InFD))
        printErrorAndExit(FD1Str + " is not a valid file descriptor");
      if (FD2Str.getAsInteger(10, OutFD))
        printErrorAndExit(FD2Str + " is not a valid file descriptor");
    } else if (SpecifierType == "listen") {
      StringRef Host, PortStr;
      std::tie(Host, PortStr) = Specifier.split(':');

      int Port = 0;
      if (PortStr.getAsInteger(10, Port))
        printErrorAndExit("port" + PortStr + " is not a valid integer");

      InFD = OutFD = openListener(Host.str(), Port);
    } else
      printErrorAndExit("invalid specifier type \"" + SpecifierType + "\"");
  }

  ExitOnErr.setBanner(std::string(argv[0]) + ":");

  using JITLinkExecutorEndpoint =
      shared::MultiThreadedRPCEndpoint<shared::FDRawByteChannel>;

  shared::registerStringError<shared::FDRawByteChannel>();

  shared::FDRawByteChannel C(InFD, OutFD);
  JITLinkExecutorEndpoint EP(C, true);
  OrcRPCTPCServer<JITLinkExecutorEndpoint> Server(EP);
  Server.setProgramName(std::string("llvm-jitlink-executor"));

  ExitOnErr(Server.run());

  return 0;
}

//===-- ExtractBundleEntryOpts.td - llvm-bitcode-strip options  ---------------*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the command line options of llvm-extract-offload-entry.
//
//===----------------------------------------------------------------------===//

include "llvm/Option/OptParser.td"

def help : Flag<["--"], "help">,
           HelpText<"URIs can be read from STDIN, one per line.\n"
                    "From the URIs specified, extracts code objects into files named:\n"
                    "\t<executable_name>-[pid<number>]-offset<number>-size<number>.co\n\n"
                    "URI syntax:\n" 
                    "\t<code_object_uri> ::== <file_uri> | <memory_uri>\n" 
                    "\t<file_uri>        ::== \"file://\" <extract_file><range_specifier>\n" 
                    "\t<memory_uri>      ::== \"memory://\"<process_id><range_specifier>\n" 
                    "\t<range_specifier> ::== [ \"#\" | \"?\" ]\"offset=\"<number>\"&size=\"<number>\n" 
                    "\t<extract_file>    ::== URI_ENCODED_OS_FILE_PATH\n" 
                    "\t<process_id>      ::== DECIMAL_NUMBER\n" 
                    "\t<number>          ::== DECIMAL_NUMBER\n" 
                    "\nExample: file://dir1/dir2/hello_world#offset=133&size=14472 \n" 
                    " memory://1234#offset=0x20000&size=3000\n">;
def h : Flag<["-"], "h">, Alias<help>;

def version : Flag<["--"], "version">,
              HelpText<"Print the version and exit">;

def V : Flag<["-"], "V">,
        Alias<version>,
        HelpText<"Alias for --version">;

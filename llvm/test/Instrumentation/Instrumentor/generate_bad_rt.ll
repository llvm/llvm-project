; RUN: not opt < %s -passes=instrumentor -instrumentor-read-config-file=%S/bad_rt_config.json 2>&1 | FileCheck %s

; CHECK: error: failed to open instrumentor stub runtime file for writing: Is a directory

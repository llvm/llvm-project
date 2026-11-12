; RUN: opt -passes=verify %S/Inputs/infer_dso_local.bc | llvm-dis | FileCheck %s

; CHECK: define linkonce_odr hidden void @test()

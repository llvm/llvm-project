; RUN: opt < %s -passes="print<cost-model>" -disable-output -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr7 -mattr=-vsx 2>&1 | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @exts() {

   ; VSX is disabled, so this cost needs to include scalarization (because
   ; <4 x double> is legalized to scalars).
   ; CHECK: cost of 44 {{.*}} fcmp
   %v1 = fcmp ugt <4 x double> undef, undef

  ret void
}


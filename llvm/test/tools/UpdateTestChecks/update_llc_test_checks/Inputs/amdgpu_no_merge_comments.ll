; RUN: llc -O0 -mtriple=amdgcn- -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9-O0 %s
; RUN: llc -mtriple=amdgcn- -mcpu=gfx900 < %s | FileCheck -check-prefixes=GCN,GFX9-O3 %s

define hidden i32 @main(i32 %a) {
  %add = add i32 %a, %a
  %mul = mul i32 %add, %a
  %sub = sub i32 %mul, %add
  ret i32 %sub
}

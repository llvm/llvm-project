; llc is replaced with cat, we just simulate llc by printing text from these files
; RUN: llc %S/amdgpu_no_merge_comments-O0.s | FileCheck -check-prefixes=GCN,GFX9-O0 %s
; RUN: llc %S/amdgpu_no_merge_comments-O3.s | FileCheck -check-prefixes=GCN,GFX9-O3 %s

target triple = "amdgcn--"

define hidden i32 @main(i32 %a) {
  %add = add i32 %a, %a
  %mul = mul i32 %add, %a
  %sub = sub i32 %mul, %add
  ret i32 %sub
}

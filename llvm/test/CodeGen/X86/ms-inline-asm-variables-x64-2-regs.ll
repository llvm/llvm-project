; RUN: not llc -mtriple=x86_64-unknown-unknown -relocation-model=pic %s -o /dev/null 2>&1 | FileCheck %s

; Tests come from "clang/test/CodeGen/ms-inline-asm-variables.c"
;
; int gVar;
; void t1() {
;  __asm add ecx, dword ptr gVar[4590 + rax + rcx*4]
;  __asm add dword ptr [gVar + rax + 45 + 23 - 53 + 60 - 2 + rcx*8], ecx
;  __asm add 1 + 1 + 2 + 3[gVar + rcx + rbx], eax
; gVar = 3;
; }

@gVar = global i32 0, align 4

; Function Attrs: noinline nounwind optnone uwtable
define void @t1() #0 {
; CHECK: error: Don't use 2 or more regs for mem offset in PIC model
; CHECK: error: Don't use 2 or more regs for mem offset in PIC model
; CHECK: error: Don't use 2 or more regs for mem offset in PIC model
entry:
  call void asm sideeffect inteldialect "add ecx, dword ptr ${2:P}[rax + rcx * $$4 + $$4590]\0A\09add dword ptr ${0:P}[rcx + rcx * $$8 + $$73], ecx\0A\09add ${1:P}[rcx + rbx + $$7], eax", "=*m,=*m,*m,~{ecx},~{flags},~{dirflag},~{fpsr},~{flags}"(ptr elementtype(i32) @gVar, ptr elementtype(i32) @gVar, ptr elementtype(i32) @gVar)
  store i32 3, ptr @gVar, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"wchar_size", i32 4}

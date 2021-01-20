; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -relocation-model=pic -post-RA-scheduler=1 -break-anti-dependencies=critical  | FileCheck %s

; PR20308 ( http://llvm.org/bugs/show_bug.cgi?id=20308 ).
; The critical-anti-dependency-breaker must not use register def information from a kill inst.
; This test case expects such an instruction to appear as a comment with def info for RDI.
; There is an anti-dependency (WAR) hazard using RAX using default reg allocation and scheduling.
; The post-RA-scheduler and critical-anti-dependency breaker can eliminate that hazard using R10.
; That is the first free register that isn't used as a param in the call to "@Image".

@PartClass = external global i32
@NullToken = external global i64

; CHECK-LABEL: Part_Create:
; CHECK: movq PartClass@GOTPCREL(%rip), %r10
define i32 @Part_Create(i64* %Anchor, i32 %TypeNum, i32 %F, i32 %Z, i32* %Status, i64* %PartTkn) {
  %PartObj = alloca i64*, align 8
  %Vchunk = alloca i64, align 8
  %1 = load i64, i64* @NullToken, align 4
  store i64 %1, i64* %Vchunk, align 8
  %2 = load i32, i32* @PartClass, align 4
  call i32 @Image(i64* %Anchor, i32 %2, i32 0, i32 0, i32* %Status, i64* %PartTkn, i64** %PartObj)
  call i32 @Create(i64* %Anchor)
  ret i32 %2
}

declare i32 @Image(i64*, i32, i32, i32, i32*, i64*, i64**)
declare i32 @Create(i64*)

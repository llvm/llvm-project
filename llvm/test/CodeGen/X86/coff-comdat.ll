; RUN: llc -mtriple i386-pc-win32 < %s | FileCheck %s

$f1 = comdat any
@v1 = global i32 0, comdat($f1)
define void @f1() comdat($f1) {
  ret void
}

$f2 = comdat exactmatch
@v2 = global i32 0, comdat($f2)
define void @f2() comdat($f2) {
  ret void
}

$f3 = comdat largest
@v3 = global i32 0, comdat($f3)
define void @f3() comdat($f3) {
  ret void
}

$f4 = comdat nodeduplicate
@v4 = global i32 0, comdat($f4)
define void @f4() comdat($f4) {
  ret void
}

$f5 = comdat samesize
@v5 = global i32 0, comdat($f5)
define void @f5() comdat($f5) {
  ret void
}

$f6 = comdat samesize
@v6 = global i32 0, comdat($f6)
@f6 = global i32 0, comdat($f6)

$"\01@f7@0" = comdat any
define x86_fastcallcc void @"\01@v7@0"() comdat($"\01@f7@0") {
  ret void
}
define x86_fastcallcc void @"\01@f7@0"() comdat($"\01@f7@0") {
  ret void
}

$f8 = comdat any
define x86_fastcallcc void @v8() comdat($f8) {
  ret void
}
define x86_fastcallcc void @f8() comdat($f8) {
  ret void
}

$vftable = comdat largest

@some_name = private unnamed_addr constant [2 x ptr] zeroinitializer, comdat($vftable)
@vftable = alias ptr, getelementptr([2 x ptr], ptr @some_name, i32 0, i32 1)

; CHECK: .section        .text,"xr",discard,_f1
; CHECK: .globl  _f1
; CHECK: .section        .text,"xr",same_contents,_f2
; CHECK: .globl  _f2
; CHECK: .section        .text,"xr",largest,_f3
; CHECK: .globl  _f3
; CHECK: .section        .text,"xr",one_only,_f4
; CHECK: .globl  _f4
; CHECK: .section        .text,"xr",same_size,_f5
; CHECK: .globl  _f5
; CHECK: .section        .text,"xr",associative,@f7@0
; CHECK: .globl  @v7@0
; CHECK: .section        .text,"xr",discard,@f7@0
; CHECK: .globl  @f7@0
; CHECK: .section        .text,"xr",associative,@f8@0
; CHECK: .globl  @v8@0
; CHECK: .section        .text,"xr",discard,@f8@0
; CHECK: .globl  @f8@0
; CHECK: .section        .bss,"bw",associative,_f1
; CHECK: .globl  _v1
; CHECK: .section        .bss,"bw",associative,_f2
; CHECK: .globl  _v2
; CHECK: .section        .bss,"bw",associative,_f3
; CHECK: .globl  _v3
; CHECK: .section        .bss,"bw",associative,_f4
; CHECK: .globl  _v4
; CHECK: .section        .bss,"bw",associative,_f5
; CHECK: .globl  _v5
; CHECK: .section        .bss,"bw",associative,_f6
; CHECK: .globl  _v6
; CHECK: .section        .bss,"bw",same_size,_f6
; CHECK: .globl  _f6
; CHECK: .section        .rdata,"dr",largest,_vftable
; CHECK: .globl  _vftable
; CHECK: .set _vftable, L_some_name+4

; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -filetype=obj < %s | llvm-readobj --codeview - | FileCheck %s --check-prefix=OBJ

; C source:
; int getval(void);
; void usevals(int, ...);
; int csr1() {
;   int a = getval();
;   usevals(a);
;   usevals(a);
;   return a;
; }
; int csr2() {
;   int a = getval();
;   int b = getval();
;   usevals(a, b);
;   usevals(a, b);
;   return a;
; }
; int csr3() {
;   int a = getval();
;   int b = getval();
;   int c = getval();
;   usevals(a, b, c);
;   usevals(a, b, c);
;   return a;
; }
; int csr4() {
;   int a = getval();
;   int b = getval();
;   int c = getval();
;   int d = getval();
;   usevals(a, b, c, d);
;   usevals(a, b, c, d);
;   return a;
; }
; int spill() {
;   int a = getval();
;   int b = getval();
;   int c = getval();
;   int d = getval();
;   int e = getval();
;   usevals(a, b, c, d, e);
;   usevals(a, b, c, d, e);
;   return a;
; }

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.25508"

; Function Attrs: nounwind
define i32 @csr1() local_unnamed_addr #0 !dbg !8 {
entry:
  %call = tail call i32 @getval() #3, !dbg !14
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !13, metadata !DIExpression()), !dbg !15
  tail call void (i32, ...) @usevals(i32 %call) #3, !dbg !16
  tail call void (i32, ...) @usevals(i32 %call) #3, !dbg !17
  ret i32 %call, !dbg !18
}

; ASM-LABEL: _csr1:                                  # @csr1
; ASM:         .cv_fpo_proc    _csr1
; ASM:         pushl   %esi
; ASM:         .cv_fpo_pushreg %esi
; ASM:         .cv_fpo_endprologue
; ASM:         #DEBUG_VALUE: csr1:a <- $esi
; ASM:         retl
; ASM:         .cv_fpo_endproc

; OBJ-LABEL: SubSectionType: FrameData (0xF5)
; OBJ-NEXT: SubSectionSize:
; OBJ-NEXT: LinkageName: _csr1
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x0
; OBJ-NEXT:   CodeSize: 0x1E
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x1
; OBJ-NEXT:   SavedRegsSize: 0x0
; OBJ-NEXT:   Flags [ (0x4)
; OBJ-NEXT:     IsFunctionStart (0x4)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x1
; OBJ-NEXT:   CodeSize: 0x1D
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x0
; OBJ-NEXT:   SavedRegsSize: 0x4
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $esi $T0 4 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NOT: FrameData

declare i32 @getval() local_unnamed_addr #1

declare void @usevals(i32, ...) local_unnamed_addr #1

; Function Attrs: nounwind
define i32 @csr2() local_unnamed_addr #0 !dbg !19 {
entry:
  %call = tail call i32 @getval() #3, !dbg !23
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !21, metadata !DIExpression()), !dbg !24
  %call1 = tail call i32 @getval() #3, !dbg !25
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !22, metadata !DIExpression()), !dbg !26
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1) #3, !dbg !27
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1) #3, !dbg !28
  ret i32 %call, !dbg !29
}

; ASM-LABEL: _csr2:                                  # @csr2
; ASM:         .cv_fpo_proc    _csr2
; ASM:         pushl   %edi
; ASM:         .cv_fpo_pushreg %edi
; ASM:         pushl   %esi
; ASM:         .cv_fpo_pushreg %esi
; ASM:         .cv_fpo_endprologue
; ASM:         #DEBUG_VALUE: csr2:a <- $esi
; ASM:         #DEBUG_VALUE: csr2:b <- $edi
; ASM:         retl
; ASM:         .cv_fpo_endproc

; OBJ-LABEL: SubSectionType: FrameData (0xF5)
; OBJ-NEXT: SubSectionSize:
; OBJ-NEXT: LinkageName: _csr2
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x0
; OBJ-NEXT:   CodeSize: 0x29
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x2
; OBJ-NEXT:   SavedRegsSize: 0x0
; OBJ-NEXT:   Flags [ (0x4)
; OBJ-NEXT:     IsFunctionStart (0x4)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x1
; OBJ-NEXT:   CodeSize: 0x28
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x1
; OBJ-NEXT:   SavedRegsSize: 0x4
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $edi $T0 4 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x2
; OBJ-NEXT:   CodeSize: 0x27
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x0
; OBJ-NEXT:   SavedRegsSize: 0x8
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $edi $T0 4 - ^ =
; OBJ-NEXT:     $esi $T0 8 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NOT: FrameData

; Function Attrs: nounwind
define i32 @csr3() local_unnamed_addr #0 !dbg !30 {
entry:
  %call = tail call i32 @getval() #3, !dbg !35
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !32, metadata !DIExpression()), !dbg !36
  %call1 = tail call i32 @getval() #3, !dbg !37
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !33, metadata !DIExpression()), !dbg !38
  %call2 = tail call i32 @getval() #3, !dbg !39
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !34, metadata !DIExpression()), !dbg !40
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1, i32 %call2) #3, !dbg !41
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1, i32 %call2) #3, !dbg !42
  ret i32 %call, !dbg !43
}

; ASM-LABEL: _csr3:                                  # @csr3
; ASM:         .cv_fpo_proc    _csr3
; ASM:         pushl   %ebx
; ASM:         .cv_fpo_pushreg %ebx
; ASM:         pushl   %edi
; ASM:         .cv_fpo_pushreg %edi
; ASM:         pushl   %esi
; ASM:         .cv_fpo_pushreg %esi
; ASM:         .cv_fpo_endprologue
; ASM:         #DEBUG_VALUE: csr3:a <- $esi
; ASM:         #DEBUG_VALUE: csr3:b <- $edi
; ASM:         #DEBUG_VALUE: csr3:c <- $ebx
; ASM:         retl
; ASM:         .cv_fpo_endproc

; OBJ-LABEL: SubSectionType: FrameData (0xF5)
; OBJ-NEXT: SubSectionSize:
; OBJ-NEXT: LinkageName: _csr3
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x0
; OBJ-NEXT:   CodeSize: 0x34
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x3
; OBJ-NEXT:   SavedRegsSize: 0x0
; OBJ-NEXT:   Flags [ (0x4)
; OBJ-NEXT:     IsFunctionStart (0x4)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x1
; OBJ-NEXT:   CodeSize: 0x33
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x2
; OBJ-NEXT:   SavedRegsSize: 0x4
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebx $T0 4 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x2
; OBJ-NEXT:   CodeSize: 0x32
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x1
; OBJ-NEXT:   SavedRegsSize: 0x8
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebx $T0 4 - ^ =
; OBJ-NEXT:     $edi $T0 8 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x3
; OBJ-NEXT:   CodeSize: 0x31
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x0
; OBJ-NEXT:   SavedRegsSize: 0xC
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebx $T0 4 - ^ =
; OBJ-NEXT:     $edi $T0 8 - ^ =
; OBJ-NEXT:     $esi $T0 12 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NOT: FrameData

; Function Attrs: nounwind
define i32 @csr4() local_unnamed_addr #0 !dbg !44 {
entry:
  %call = tail call i32 @getval() #3, !dbg !50
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !46, metadata !DIExpression()), !dbg !51
  %call1 = tail call i32 @getval() #3, !dbg !52
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !47, metadata !DIExpression()), !dbg !53
  %call2 = tail call i32 @getval() #3, !dbg !54
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !48, metadata !DIExpression()), !dbg !55
  %call3 = tail call i32 @getval() #3, !dbg !56
  tail call void @llvm.dbg.value(metadata i32 %call3, metadata !49, metadata !DIExpression()), !dbg !57
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1, i32 %call2, i32 %call3) #3, !dbg !58
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1, i32 %call2, i32 %call3) #3, !dbg !59
  ret i32 %call, !dbg !60
}

; ASM-LABEL: _csr4:                                  # @csr4
; ASM:         .cv_fpo_proc    _csr4
; ASM:         pushl   %ebp
; ASM:         .cv_fpo_pushreg %ebp
; ASM:         pushl   %ebx
; ASM:         .cv_fpo_pushreg %ebx
; ASM:         pushl   %edi
; ASM:         .cv_fpo_pushreg %edi
; ASM:         pushl   %esi
; ASM:         .cv_fpo_pushreg %esi
; ASM:         .cv_fpo_endprologue
; ASM:         #DEBUG_VALUE: csr4:a <- $esi
; ASM:         #DEBUG_VALUE: csr4:b <- $edi
; ASM:         #DEBUG_VALUE: csr4:c <- $ebx
; ASM:         #DEBUG_VALUE: csr4:d <- $ebp
; ASM:         retl
; ASM:         .cv_fpo_endproc

; OBJ-LABEL: SubSectionType: FrameData (0xF5)
; OBJ-NEXT: SubSectionSize:
; OBJ-NEXT: LinkageName: _csr4
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x0
; OBJ-NEXT:   CodeSize: 0x3F
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x4
; OBJ-NEXT:   SavedRegsSize: 0x0
; OBJ-NEXT:   Flags [ (0x4)
; OBJ-NEXT:     IsFunctionStart (0x4)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x1
; OBJ-NEXT:   CodeSize: 0x3E
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x3
; OBJ-NEXT:   SavedRegsSize: 0x4
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x2
; OBJ-NEXT:   CodeSize: 0x3D
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x2
; OBJ-NEXT:   SavedRegsSize: 0x8
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x3
; OBJ-NEXT:   CodeSize: 0x3C
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x1
; OBJ-NEXT:   SavedRegsSize: 0xC
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:     $edi $T0 12 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x4
; OBJ-NEXT:   CodeSize: 0x3B
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x0
; OBJ-NEXT:   SavedRegsSize: 0x10
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:     $edi $T0 12 - ^ =
; OBJ-NEXT:     $esi $T0 16 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NOT: FrameData

; Function Attrs: nounwind
define i32 @spill() local_unnamed_addr #0 !dbg !61 {
entry:
  %call = tail call i32 @getval() #3, !dbg !68
  tail call void @llvm.dbg.value(metadata i32 %call, metadata !63, metadata !DIExpression()), !dbg !69
  %call1 = tail call i32 @getval() #3, !dbg !70
  tail call void @llvm.dbg.value(metadata i32 %call1, metadata !64, metadata !DIExpression()), !dbg !71
  %call2 = tail call i32 @getval() #3, !dbg !72
  tail call void @llvm.dbg.value(metadata i32 %call2, metadata !65, metadata !DIExpression()), !dbg !73
  %call3 = tail call i32 @getval() #3, !dbg !74
  tail call void @llvm.dbg.value(metadata i32 %call3, metadata !66, metadata !DIExpression()), !dbg !75
  %call4 = tail call i32 @getval() #3, !dbg !76
  tail call void @llvm.dbg.value(metadata i32 %call4, metadata !67, metadata !DIExpression()), !dbg !77
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1, i32 %call2, i32 %call3, i32 %call4) #3, !dbg !78
  tail call void (i32, ...) @usevals(i32 %call, i32 %call1, i32 %call2, i32 %call3, i32 %call4) #3, !dbg !79
  ret i32 %call, !dbg !80
}

; ASM-LABEL: _spill:                                  # @spill
; ASM:         .cv_fpo_proc    _spill
; ASM:         pushl   %ebp
; ASM:         .cv_fpo_pushreg %ebp
; ASM:         pushl   %ebx
; ASM:         .cv_fpo_pushreg %ebx
; ASM:         pushl   %edi
; ASM:         .cv_fpo_pushreg %edi
; ASM:         pushl   %esi
; ASM:         .cv_fpo_pushreg %esi
; ASM:         subl    $8, %esp
; ASM:         .cv_fpo_stackalloc 8
; ASM:         .cv_fpo_endprologue
; ASM:         retl
; ASM:         .cv_fpo_endproc

; OBJ-LABEL: SubSectionType: FrameData (0xF5)
; OBJ-NEXT: SubSectionSize:
; OBJ-NEXT: LinkageName: _spill
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x0
; OBJ-NEXT:   CodeSize: 0x5A
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x7
; OBJ-NEXT:   SavedRegsSize: 0x0
; OBJ-NEXT:   Flags [ (0x4)
; OBJ-NEXT:     IsFunctionStart (0x4)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x1
; OBJ-NEXT:   CodeSize: 0x59
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x6
; OBJ-NEXT:   SavedRegsSize: 0x4
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x2
; OBJ-NEXT:   CodeSize: 0x58
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x5
; OBJ-NEXT:   SavedRegsSize: 0x8
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x3
; OBJ-NEXT:   CodeSize: 0x57
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x4
; OBJ-NEXT:   SavedRegsSize: 0xC
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:     $edi $T0 12 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x4
; OBJ-NEXT:   CodeSize: 0x56
; OBJ-NEXT:   LocalSize: 0x0
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x3
; OBJ-NEXT:   SavedRegsSize: 0x10
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:     $edi $T0 12 - ^ =
; OBJ-NEXT:     $esi $T0 16 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NEXT: FrameData {
; OBJ-NEXT:   RvaStart: 0x7
; OBJ-NEXT:   CodeSize: 0x53
; OBJ-NEXT:   LocalSize: 0x8
; OBJ-NEXT:   ParamsSize: 0x0
; OBJ-NEXT:   MaxStackSize: 0x0
; OBJ-NEXT:   PrologSize: 0x0
; OBJ-NEXT:   SavedRegsSize: 0x10
; OBJ-NEXT:   Flags [ (0x0)
; OBJ-NEXT:   ]
; OBJ-NEXT:   FrameFunc [
; OBJ-NEXT:     $T0 .raSearch =
; OBJ-NEXT:     $eip $T0 ^ =
; OBJ-NEXT:     $esp $T0 4 + =
; OBJ-NEXT:     $ebp $T0 4 - ^ =
; OBJ-NEXT:     $ebx $T0 8 - ^ =
; OBJ-NEXT:     $edi $T0 12 - ^ =
; OBJ-NEXT:     $esi $T0 16 - ^ =
; OBJ-NEXT:   ]
; OBJ-NEXT: }
; OBJ-NOT: FrameData

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone speculatable }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 6.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "0b1c85f8a0bfb41380df1fcaeadde306")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "csr1", scope: !1, file: !1, line: 3, type: !9, isLocal: false, isDefinition: true, scopeLine: 3, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 4, type: !11)
!14 = !DILocation(line: 4, column: 11, scope: !8)
!15 = !DILocation(line: 4, column: 7, scope: !8)
!16 = !DILocation(line: 5, column: 3, scope: !8)
!17 = !DILocation(line: 6, column: 3, scope: !8)
!18 = !DILocation(line: 7, column: 3, scope: !8)
!19 = distinct !DISubprogram(name: "csr2", scope: !1, file: !1, line: 9, type: !9, isLocal: false, isDefinition: true, scopeLine: 9, isOptimized: true, unit: !0, retainedNodes: !20)
!20 = !{!21, !22}
!21 = !DILocalVariable(name: "a", scope: !19, file: !1, line: 10, type: !11)
!22 = !DILocalVariable(name: "b", scope: !19, file: !1, line: 11, type: !11)
!23 = !DILocation(line: 10, column: 11, scope: !19)
!24 = !DILocation(line: 10, column: 7, scope: !19)
!25 = !DILocation(line: 11, column: 11, scope: !19)
!26 = !DILocation(line: 11, column: 7, scope: !19)
!27 = !DILocation(line: 12, column: 3, scope: !19)
!28 = !DILocation(line: 13, column: 3, scope: !19)
!29 = !DILocation(line: 14, column: 3, scope: !19)
!30 = distinct !DISubprogram(name: "csr3", scope: !1, file: !1, line: 16, type: !9, isLocal: false, isDefinition: true, scopeLine: 16, isOptimized: true, unit: !0, retainedNodes: !31)
!31 = !{!32, !33, !34}
!32 = !DILocalVariable(name: "a", scope: !30, file: !1, line: 17, type: !11)
!33 = !DILocalVariable(name: "b", scope: !30, file: !1, line: 18, type: !11)
!34 = !DILocalVariable(name: "c", scope: !30, file: !1, line: 19, type: !11)
!35 = !DILocation(line: 17, column: 11, scope: !30)
!36 = !DILocation(line: 17, column: 7, scope: !30)
!37 = !DILocation(line: 18, column: 11, scope: !30)
!38 = !DILocation(line: 18, column: 7, scope: !30)
!39 = !DILocation(line: 19, column: 11, scope: !30)
!40 = !DILocation(line: 19, column: 7, scope: !30)
!41 = !DILocation(line: 20, column: 3, scope: !30)
!42 = !DILocation(line: 21, column: 3, scope: !30)
!43 = !DILocation(line: 22, column: 3, scope: !30)
!44 = distinct !DISubprogram(name: "csr4", scope: !1, file: !1, line: 24, type: !9, isLocal: false, isDefinition: true, scopeLine: 24, isOptimized: true, unit: !0, retainedNodes: !45)
!45 = !{!46, !47, !48, !49}
!46 = !DILocalVariable(name: "a", scope: !44, file: !1, line: 25, type: !11)
!47 = !DILocalVariable(name: "b", scope: !44, file: !1, line: 26, type: !11)
!48 = !DILocalVariable(name: "c", scope: !44, file: !1, line: 27, type: !11)
!49 = !DILocalVariable(name: "d", scope: !44, file: !1, line: 28, type: !11)
!50 = !DILocation(line: 25, column: 11, scope: !44)
!51 = !DILocation(line: 25, column: 7, scope: !44)
!52 = !DILocation(line: 26, column: 11, scope: !44)
!53 = !DILocation(line: 26, column: 7, scope: !44)
!54 = !DILocation(line: 27, column: 11, scope: !44)
!55 = !DILocation(line: 27, column: 7, scope: !44)
!56 = !DILocation(line: 28, column: 11, scope: !44)
!57 = !DILocation(line: 28, column: 7, scope: !44)
!58 = !DILocation(line: 29, column: 3, scope: !44)
!59 = !DILocation(line: 30, column: 3, scope: !44)
!60 = !DILocation(line: 31, column: 3, scope: !44)
!61 = distinct !DISubprogram(name: "spill", scope: !1, file: !1, line: 33, type: !9, isLocal: false, isDefinition: true, scopeLine: 33, isOptimized: true, unit: !0, retainedNodes: !62)
!62 = !{!63, !64, !65, !66, !67}
!63 = !DILocalVariable(name: "a", scope: !61, file: !1, line: 34, type: !11)
!64 = !DILocalVariable(name: "b", scope: !61, file: !1, line: 35, type: !11)
!65 = !DILocalVariable(name: "c", scope: !61, file: !1, line: 36, type: !11)
!66 = !DILocalVariable(name: "d", scope: !61, file: !1, line: 37, type: !11)
!67 = !DILocalVariable(name: "e", scope: !61, file: !1, line: 38, type: !11)
!68 = !DILocation(line: 34, column: 11, scope: !61)
!69 = !DILocation(line: 34, column: 7, scope: !61)
!70 = !DILocation(line: 35, column: 11, scope: !61)
!71 = !DILocation(line: 35, column: 7, scope: !61)
!72 = !DILocation(line: 36, column: 11, scope: !61)
!73 = !DILocation(line: 36, column: 7, scope: !61)
!74 = !DILocation(line: 37, column: 11, scope: !61)
!75 = !DILocation(line: 37, column: 7, scope: !61)
!76 = !DILocation(line: 38, column: 11, scope: !61)
!77 = !DILocation(line: 38, column: 7, scope: !61)
!78 = !DILocation(line: 39, column: 3, scope: !61)
!79 = !DILocation(line: 40, column: 3, scope: !61)
!80 = !DILocation(line: 41, column: 3, scope: !61)

; RUN: llc -filetype=obj %s -o %t.obj
; RUN: llvm-pdbutil dump %t.obj -symbols | FileCheck %s

; A fairly exhaustive test of S_FRAMEPROC flags. Use the source below to compare
; the flags we set with MSVC.

; extern "C" {
;
; void *_alloca(size_t);
; struct __declspec(align(16)) _jmp_buf_str {
;   unsigned __int64 Part[2];
; };
; typedef struct _jmp_buf_str jmp_buf[16];
; int __cdecl _setjmp(jmp_buf _Buf);
;
; void may_throw(void);
; void use_intptr(int *);
;
; void use_alloca(int n) {
;   int *p = (int*)_alloca(n * sizeof(int));
;   use_intptr(p);
; }
;
; jmp_buf g_jbuf;
; void call_setjmp(int n) {
;   if (!_setjmp(g_jbuf))
;     use_intptr(nullptr);
; }
;
; void use_inlineasm() {
;   __asm nop
; }
;
; void cpp_eh() {
;   try {
;     may_throw();
;   } catch (...) {
;   }
; }
;
; static inline int is_marked_inline(int x, int y) {
;   return x + y;
; }
; int (*use_inline())(int x, int y) {
;   return &is_marked_inline;
; }
;
; void seh() {
;   __try {
;     may_throw();
;   } __except (1) {
;   }
; }
;
; void __declspec(naked) use_naked() {
;   __asm ret
; }
;
; void stack_guard() {
;   int arr[12] = {0};
;   use_intptr(&arr[0]);
; }
; }

; CHECK-LABEL: S_GPROC32_ID [size = 52] `use_alloca`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = VFRAME, param fp reg = EBP
; CHECK:   flags = has alloca | secure checks | opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 52] `call_setjmp`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = NONE, param fp reg = NONE
; CHECK:   flags = has setjmp | opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 56] `use_inlineasm`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = NONE, param fp reg = NONE
; CHECK:   flags = has inline asm | opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 48] `cpp_eh`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = EBP, param fp reg = EBP
; CHECK:   flags = has eh | opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 52] `use_inline`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = NONE, param fp reg = NONE
; CHECK:   flags = opt speed
; CHECK-LABEL: S_LPROC32_ID [size = 56] `is_marked_inline`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = NONE, param fp reg = NONE
; CHECK:   flags = marked inline | opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 44] `seh`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = EBP, param fp reg = EBP
; CHECK:   flags = has seh | opt speed
; CHECK-LABEL: S_LPROC32_ID [size = 56] `?filt$0@0@seh@@`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = EBP, param fp reg = EBP
; CHECK:   flags = opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 52] `use_naked`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = NONE, param fp reg = NONE
; CHECK:   flags = has inline asm | naked | opt speed
; CHECK-LABEL: S_GPROC32_ID [size = 52] `stack_guard`
; CHECK: S_FRAMEPROC [size = 32]
; CHECK:   local fp reg = VFRAME, param fp reg = EBP
; CHECK:   flags = secure checks | opt speed

; ModuleID = 'frameproc-flags.cpp'
source_filename = "frameproc-flags.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.14.26433"

%struct._jmp_buf_str = type { [2 x i64] }

@g_jbuf = dso_local global [16 x %struct._jmp_buf_str] zeroinitializer, align 16, !dbg !0

define dso_local void @use_alloca(i32 %n) local_unnamed_addr #0 !dbg !25 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !29, metadata !DIExpression()), !dbg !31
  %mul = shl i32 %n, 2, !dbg !32
  %0 = alloca i8, i32 %mul, align 16, !dbg !32
  %1 = bitcast i8* %0 to i32*, !dbg !32
  call void @llvm.dbg.value(metadata i32* %1, metadata !30, metadata !DIExpression()), !dbg !32
  call void @use_intptr(i32* nonnull %1), !dbg !33
  ret void, !dbg !34
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) #2

declare dso_local void @use_intptr(i32*) local_unnamed_addr #3

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) #2

define dso_local void @call_setjmp(i32 %n) local_unnamed_addr #0 !dbg !35 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !37, metadata !DIExpression()), !dbg !38
  %0 = call i32 (i8*, i32, ...) @_setjmp3(i8* bitcast ([16 x %struct._jmp_buf_str]* @g_jbuf to i8*), i32 0) #4, !dbg !39
  %tobool = icmp eq i32 %0, 0, !dbg !39
  br i1 %tobool, label %if.then, label %if.end, !dbg !39

if.then:                                          ; preds = %entry
  call void @use_intptr(i32* null), !dbg !40
  br label %if.end, !dbg !40

if.end:                                           ; preds = %entry, %if.then
  ret void, !dbg !42
}

; Function Attrs: returns_twice
declare dso_local i32 @_setjmp3(i8*, i32, ...) local_unnamed_addr #4

; Function Attrs: nounwind
define dso_local void @use_inlineasm() local_unnamed_addr #5 !dbg !43 {
entry:
  tail call void asm sideeffect inteldialect "nop", "~{dirflag},~{fpsr},~{flags}"() #10, !dbg !46, !srcloc !47
  ret void, !dbg !48
}

define dso_local void @cpp_eh() local_unnamed_addr #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) !dbg !49 {
entry:
  invoke void @may_throw()
          to label %try.cont unwind label %catch.dispatch, !dbg !50

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind to caller, !dbg !52

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null], !dbg !52
  catchret from %1 to label %try.cont, !dbg !53

try.cont:                                         ; preds = %entry, %catch
  ret void, !dbg !55
}

declare dso_local void @may_throw() local_unnamed_addr #3

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: norecurse nounwind readnone
define dso_local nonnull i32 (i32, i32)* @use_inline() local_unnamed_addr #6 !dbg !56 {
entry:
  ret i32 (i32, i32)* @"?is_marked_inline@@YAHHH@Z", !dbg !62
}

; Function Attrs: inlinehint nounwind readnone
define internal i32 @"?is_marked_inline@@YAHHH@Z"(i32 %x, i32 %y) #7 !dbg !63 {
entry:
  call void @llvm.dbg.value(metadata i32 %y, metadata !65, metadata !DIExpression()), !dbg !67
  call void @llvm.dbg.value(metadata i32 %x, metadata !66, metadata !DIExpression()), !dbg !67
  %add = add nsw i32 %y, %x, !dbg !68
  ret i32 %add, !dbg !68
}

define dso_local void @seh() #0 personality i8* bitcast (i32 (...)* @_except_handler3 to i8*) !dbg !69 {
entry:
  %__exception_code = alloca i32, align 4
  call void (...) @llvm.localescape(i32* nonnull %__exception_code)
  invoke void @may_throw() #12
          to label %__try.cont unwind label %catch.dispatch, !dbg !70

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %__except.ret] unwind to caller, !dbg !72

__except.ret:                                     ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i32 ()* @"?filt$0@0@seh@@" to i8*)], !dbg !72
  catchret from %1 to label %__try.cont, !dbg !72

__try.cont:                                       ; preds = %entry, %__except.ret
  ret void, !dbg !73
}

; Function Attrs: nounwind
define internal i32 @"?filt$0@0@seh@@"() #8 !dbg !74 {
entry:
  %0 = tail call i8* @llvm.frameaddress(i32 1)
  %1 = tail call i8* @llvm.eh.recoverfp(i8* bitcast (void ()* @seh to i8*), i8* %0)
  %2 = tail call i8* @llvm.localrecover(i8* bitcast (void ()* @seh to i8*), i8* %1, i32 0)
  %__exception_code = bitcast i8* %2 to i32*
  %3 = getelementptr inbounds i8, i8* %0, i32 -20, !dbg !76
  %4 = bitcast i8* %3 to { i32*, i8* }**, !dbg !76
  %5 = load { i32*, i8* }*, { i32*, i8* }** %4, align 4, !dbg !76
  %6 = getelementptr inbounds { i32*, i8* }, { i32*, i8* }* %5, i32 0, i32 0, !dbg !76
  %7 = load i32*, i32** %6, align 4, !dbg !76
  %8 = load i32, i32* %7, align 4, !dbg !76
  store i32 %8, i32* %__exception_code, align 4, !dbg !76
  ret i32 1, !dbg !76
}

; Function Attrs: nounwind readnone
declare i8* @llvm.frameaddress(i32) #9

; Function Attrs: nounwind readnone
declare i8* @llvm.eh.recoverfp(i8*, i8*) #9

; Function Attrs: nounwind readnone
declare i8* @llvm.localrecover(i8*, i8*, i32) #9

declare dso_local i32 @_except_handler3(...)

; Function Attrs: nounwind
declare void @llvm.localescape(...) #10

; Function Attrs: naked noinline nounwind
define dso_local void @use_naked() #11 !dbg !77 {
entry:
  tail call void asm sideeffect inteldialect "ret", "~{dirflag},~{fpsr},~{flags}"() #10, !dbg !78, !srcloc !79
  unreachable, !dbg !80
}

define dso_local void @stack_guard() local_unnamed_addr #0 !dbg !81 {
entry:
  %arr = alloca [12 x i32], align 4
  %0 = bitcast [12 x i32]* %arr to i8*, !dbg !87
  call void @llvm.lifetime.start.p0i8(i64 48, i8* nonnull %0) #10, !dbg !87
  call void @llvm.dbg.declare(metadata [12 x i32]* %arr, metadata !83, metadata !DIExpression()), !dbg !87
  call void @llvm.memset.p0i8.i32(i8* nonnull align 4 %0, i8 0, i32 48, i1 false), !dbg !87
  %arrayidx = getelementptr inbounds [12 x i32], [12 x i32]* %arr, i32 0, i32 0, !dbg !88
  call void @use_intptr(i32* nonnull %arrayidx), !dbg !88
  call void @llvm.lifetime.end.p0i8(i64 48, i8* nonnull %0) #10, !dbg !89
  ret void, !dbg !89
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #2

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { sspstrong "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { argmemonly nounwind }
attributes #3 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { returns_twice }
attributes #5 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { inlinehint nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #8 = { nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #9 = { nounwind readnone }
attributes #10 = { nounwind }
attributes #11 = { naked noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #12 = { noinline }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!20, !21, !22, !23}
!llvm.ident = !{!24}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "g_jbuf", scope: !2, file: !3, line: 18, type: !9, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "clang version 8.0.0 ", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !8, nameTableKind: None)
!3 = !DIFile(filename: "frameproc-flags.cpp", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "1dd66a71668512c95552767c3a35300a")
!4 = !{}
!5 = !{!6}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!8 = !{!0}
!9 = !DIDerivedType(tag: DW_TAG_typedef, name: "jmp_buf", file: !3, line: 7, baseType: !10)
!10 = !DICompositeType(tag: DW_TAG_array_type, baseType: !11, size: 2048, elements: !18)
!11 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "_jmp_buf_str", file: !3, line: 4, size: 128, align: 128, flags: DIFlagTypePassByValue, elements: !12, identifier: ".?AU_jmp_buf_str@@")
!12 = !{!13}
!13 = !DIDerivedType(tag: DW_TAG_member, name: "Part", scope: !11, file: !3, line: 5, baseType: !14, size: 128)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !15, size: 128, elements: !16)
!15 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!16 = !{!17}
!17 = !DISubrange(count: 2)
!18 = !{!19}
!19 = !DISubrange(count: 16)
!20 = !{i32 1, !"NumRegisterParameters", i32 0}
!21 = !{i32 2, !"CodeView", i32 1}
!22 = !{i32 2, !"Debug Info Version", i32 3}
!23 = !{i32 1, !"wchar_size", i32 2}
!24 = !{!"clang version 8.0.0 "}
!25 = distinct !DISubprogram(name: "use_alloca", scope: !3, file: !3, line: 13, type: !26, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !28)
!26 = !DISubroutineType(types: !27)
!27 = !{null, !7}
!28 = !{!29, !30}
!29 = !DILocalVariable(name: "n", arg: 1, scope: !25, file: !3, line: 13, type: !7)
!30 = !DILocalVariable(name: "p", scope: !25, file: !3, line: 14, type: !6)
!31 = !DILocation(line: 13, scope: !25)
!32 = !DILocation(line: 14, scope: !25)
!33 = !DILocation(line: 15, scope: !25)
!34 = !DILocation(line: 16, scope: !25)
!35 = distinct !DISubprogram(name: "call_setjmp", scope: !3, file: !3, line: 19, type: !26, isLocal: false, isDefinition: true, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !36)
!36 = !{!37}
!37 = !DILocalVariable(name: "n", arg: 1, scope: !35, file: !3, line: 19, type: !7)
!38 = !DILocation(line: 19, scope: !35)
!39 = !DILocation(line: 20, scope: !35)
!40 = !DILocation(line: 21, scope: !41)
!41 = distinct !DILexicalBlock(scope: !35, file: !3, line: 20)
!42 = !DILocation(line: 22, scope: !35)
!43 = distinct !DISubprogram(name: "use_inlineasm", scope: !3, file: !3, line: 24, type: !44, isLocal: false, isDefinition: true, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!44 = !DISubroutineType(types: !45)
!45 = !{null}
!46 = !DILocation(line: 25, scope: !43)
!47 = !{i32 445}
!48 = !DILocation(line: 26, scope: !43)
!49 = distinct !DISubprogram(name: "cpp_eh", scope: !3, file: !3, line: 28, type: !44, isLocal: false, isDefinition: true, scopeLine: 28, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!50 = !DILocation(line: 30, scope: !51)
!51 = distinct !DILexicalBlock(scope: !49, file: !3, line: 29)
!52 = !DILocation(line: 31, scope: !51)
!53 = !DILocation(line: 32, scope: !54)
!54 = distinct !DILexicalBlock(scope: !49, file: !3, line: 31)
!55 = !DILocation(line: 33, scope: !49)
!56 = distinct !DISubprogram(name: "use_inline", scope: !3, file: !3, line: 38, type: !57, isLocal: false, isDefinition: true, scopeLine: 38, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!57 = !DISubroutineType(types: !58)
!58 = !{!59}
!59 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !60, size: 32)
!60 = !DISubroutineType(types: !61)
!61 = !{!7, !7, !7}
!62 = !DILocation(line: 39, scope: !56)
!63 = distinct !DISubprogram(name: "is_marked_inline", linkageName: "?is_marked_inline@@YAHHH@Z", scope: !3, file: !3, line: 35, type: !60, isLocal: true, isDefinition: true, scopeLine: 35, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !64)
!64 = !{!65, !66}
!65 = !DILocalVariable(name: "y", arg: 2, scope: !63, file: !3, line: 35, type: !7)
!66 = !DILocalVariable(name: "x", arg: 1, scope: !63, file: !3, line: 35, type: !7)
!67 = !DILocation(line: 35, scope: !63)
!68 = !DILocation(line: 36, scope: !63)
!69 = distinct !DISubprogram(name: "seh", scope: !3, file: !3, line: 42, type: !44, isLocal: false, isDefinition: true, scopeLine: 42, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!70 = !DILocation(line: 44, scope: !71)
!71 = distinct !DILexicalBlock(scope: !69, file: !3, line: 43)
!72 = !DILocation(line: 45, scope: !71)
!73 = !DILocation(line: 47, scope: !69)
!74 = distinct !DISubprogram(linkageName: "?filt$0@0@seh@@", scope: !3, file: !3, line: 45, type: !75, isLocal: true, isDefinition: true, scopeLine: 45, flags: DIFlagArtificial, isOptimized: true, unit: !2, retainedNodes: !4)
!75 = !DISubroutineType(types: !4)
!76 = !DILocation(line: 45, scope: !74)
!77 = distinct !DISubprogram(name: "use_naked", scope: !3, file: !3, line: 49, type: !44, isLocal: false, isDefinition: true, scopeLine: 49, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !4)
!78 = !DILocation(line: 50, scope: !77)
!79 = !{i32 765}
!80 = !DILocation(line: 51, scope: !77)
!81 = distinct !DISubprogram(name: "stack_guard", scope: !3, file: !3, line: 53, type: !44, isLocal: false, isDefinition: true, scopeLine: 53, flags: DIFlagPrototyped, isOptimized: true, unit: !2, retainedNodes: !82)
!82 = !{!83}
!83 = !DILocalVariable(name: "arr", scope: !81, file: !3, line: 54, type: !84)
!84 = !DICompositeType(tag: DW_TAG_array_type, baseType: !7, size: 384, elements: !85)
!85 = !{!86}
!86 = !DISubrange(count: 12)
!87 = !DILocation(line: 54, scope: !81)
!88 = !DILocation(line: 55, scope: !81)
!89 = !DILocation(line: 56, scope: !81)

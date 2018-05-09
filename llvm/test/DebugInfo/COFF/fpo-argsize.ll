; RUN: llc < %s | grep cv_fpo_proc | FileCheck %s

; C++ source:
; extern "C" {
; extern int g;
; 
; void cdecl1(int a) { g += a; }
; void cdecl2(int a, int b) { g += a + b; }
; void cdecl3(int a, int b, int c) { g += a + b + c; }
; 
; void __fastcall fastcall1(int a) { g += a; }
; void __fastcall fastcall2(int a, int b) { g += a + b; }
; void __fastcall fastcall3(int a, int b, int c) { g += a + b + c; }
; 
; void __stdcall stdcall1(int a) { g += a; }
; void __stdcall stdcall2(int a, int b) { g += a + b; }
; void __stdcall stdcall3(int a, int b, int c) { g += a + b + c; }
; }
; 
; struct Foo {
;   void thiscall1(int a);
;   void thiscall2(int a, int b);
;   void thiscall3(int a, int b, int c);
; };
; 
; void Foo::thiscall1(int a) { g += a; }
; void Foo::thiscall2(int a, int b) { g += a + b; }
; void Foo::thiscall3(int a, int b, int c) { g += a + b + c; }

; CHECK: .cv_fpo_proc    _cdecl1 4
; CHECK: .cv_fpo_proc    _cdecl2 8
; CHECK: .cv_fpo_proc    _cdecl3 12

; First two args are in registers and don't count.
; CHECK: .cv_fpo_proc    @fastcall1@4 0
; CHECK: .cv_fpo_proc    @fastcall2@8 0
; CHECK: .cv_fpo_proc    @fastcall3@12 4

; CHECK: .cv_fpo_proc    _stdcall1@4 4
; CHECK: .cv_fpo_proc    _stdcall2@8 8
; CHECK: .cv_fpo_proc    _stdcall3@12 12

; 'this' is in ecx and doesn't count.
; CHECK: .cv_fpo_proc    "?thiscall1@Foo@@QAEXH@Z" 4
; CHECK: .cv_fpo_proc    "?thiscall2@Foo@@QAEXHH@Z" 8
; CHECK: .cv_fpo_proc    "?thiscall3@Foo@@QAEXHHH@Z" 12

; ModuleID = 't.c'
source_filename = "t.c"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc19.11.25508"

%struct.Foo = type { i8 }

@g = external global i32, align 4

; Function Attrs: noinline nounwind optnone
define void @cdecl1(i32 %a) #0 !dbg !8 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !12, metadata !DIExpression()), !dbg !13
  %0 = load i32, i32* %a.addr, align 4, !dbg !14
  %1 = load i32, i32* @g, align 4, !dbg !15
  %add = add nsw i32 %1, %0, !dbg !15
  store i32 %add, i32* @g, align 4, !dbg !15
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone
define void @cdecl2(i32 %a, i32 %b) #0 !dbg !17 {
entry:
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !20, metadata !DIExpression()), !dbg !21
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !22, metadata !DIExpression()), !dbg !23
  %0 = load i32, i32* %a.addr, align 4, !dbg !24
  %1 = load i32, i32* %b.addr, align 4, !dbg !25
  %add = add nsw i32 %0, %1, !dbg !26
  %2 = load i32, i32* @g, align 4, !dbg !27
  %add1 = add nsw i32 %2, %add, !dbg !27
  store i32 %add1, i32* @g, align 4, !dbg !27
  ret void, !dbg !28
}

; Function Attrs: noinline nounwind optnone
define void @cdecl3(i32 %a, i32 %b, i32 %c) #0 !dbg !29 {
entry:
  %c.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !32, metadata !DIExpression()), !dbg !33
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !34, metadata !DIExpression()), !dbg !35
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !36, metadata !DIExpression()), !dbg !37
  %0 = load i32, i32* %a.addr, align 4, !dbg !38
  %1 = load i32, i32* %b.addr, align 4, !dbg !39
  %add = add nsw i32 %0, %1, !dbg !40
  %2 = load i32, i32* %c.addr, align 4, !dbg !41
  %add1 = add nsw i32 %add, %2, !dbg !42
  %3 = load i32, i32* @g, align 4, !dbg !43
  %add2 = add nsw i32 %3, %add1, !dbg !43
  store i32 %add2, i32* @g, align 4, !dbg !43
  ret void, !dbg !44
}

; Function Attrs: noinline nounwind optnone
define x86_fastcallcc void @"\01@fastcall1@4"(i32 inreg %a) #0 !dbg !45 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !47, metadata !DIExpression()), !dbg !48
  %0 = load i32, i32* %a.addr, align 4, !dbg !49
  %1 = load i32, i32* @g, align 4, !dbg !50
  %add = add nsw i32 %1, %0, !dbg !50
  store i32 %add, i32* @g, align 4, !dbg !50
  ret void, !dbg !51
}

; Function Attrs: noinline nounwind optnone
define x86_fastcallcc void @"\01@fastcall2@8"(i32 inreg %a, i32 inreg %b) #0 !dbg !52 {
entry:
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !54, metadata !DIExpression()), !dbg !55
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !56, metadata !DIExpression()), !dbg !57
  %0 = load i32, i32* %a.addr, align 4, !dbg !58
  %1 = load i32, i32* %b.addr, align 4, !dbg !59
  %add = add nsw i32 %0, %1, !dbg !60
  %2 = load i32, i32* @g, align 4, !dbg !61
  %add1 = add nsw i32 %2, %add, !dbg !61
  store i32 %add1, i32* @g, align 4, !dbg !61
  ret void, !dbg !62
}

; Function Attrs: noinline nounwind optnone
define x86_fastcallcc void @"\01@fastcall3@12"(i32 inreg %a, i32 inreg %b, i32 %c) #0 !dbg !63 {
entry:
  %c.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !65, metadata !DIExpression()), !dbg !66
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !67, metadata !DIExpression()), !dbg !68
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !69, metadata !DIExpression()), !dbg !70
  %0 = load i32, i32* %a.addr, align 4, !dbg !71
  %1 = load i32, i32* %b.addr, align 4, !dbg !72
  %add = add nsw i32 %0, %1, !dbg !73
  %2 = load i32, i32* %c.addr, align 4, !dbg !74
  %add1 = add nsw i32 %add, %2, !dbg !75
  %3 = load i32, i32* @g, align 4, !dbg !76
  %add2 = add nsw i32 %3, %add1, !dbg !76
  store i32 %add2, i32* @g, align 4, !dbg !76
  ret void, !dbg !77
}

; Function Attrs: noinline nounwind optnone
define x86_stdcallcc void @"\01_stdcall1@4"(i32 %a) #0 !dbg !78 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !80, metadata !DIExpression()), !dbg !81
  %0 = load i32, i32* %a.addr, align 4, !dbg !82
  %1 = load i32, i32* @g, align 4, !dbg !83
  %add = add nsw i32 %1, %0, !dbg !83
  store i32 %add, i32* @g, align 4, !dbg !83
  ret void, !dbg !84
}

; Function Attrs: noinline nounwind optnone
define x86_stdcallcc void @"\01_stdcall2@8"(i32 %a, i32 %b) #0 !dbg !85 {
entry:
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !87, metadata !DIExpression()), !dbg !88
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !89, metadata !DIExpression()), !dbg !90
  %0 = load i32, i32* %a.addr, align 4, !dbg !91
  %1 = load i32, i32* %b.addr, align 4, !dbg !92
  %add = add nsw i32 %0, %1, !dbg !93
  %2 = load i32, i32* @g, align 4, !dbg !94
  %add1 = add nsw i32 %2, %add, !dbg !94
  store i32 %add1, i32* @g, align 4, !dbg !94
  ret void, !dbg !95
}

; Function Attrs: noinline nounwind optnone
define x86_stdcallcc void @"\01_stdcall3@12"(i32 %a, i32 %b, i32 %c) #0 !dbg !96 {
entry:
  %c.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !98, metadata !DIExpression()), !dbg !99
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !100, metadata !DIExpression()), !dbg !101
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !102, metadata !DIExpression()), !dbg !103
  %0 = load i32, i32* %a.addr, align 4, !dbg !104
  %1 = load i32, i32* %b.addr, align 4, !dbg !105
  %add = add nsw i32 %0, %1, !dbg !106
  %2 = load i32, i32* %c.addr, align 4, !dbg !107
  %add1 = add nsw i32 %add, %2, !dbg !108
  %3 = load i32, i32* @g, align 4, !dbg !109
  %add2 = add nsw i32 %3, %add1, !dbg !109
  store i32 %add2, i32* @g, align 4, !dbg !109
  ret void, !dbg !110
}

; Function Attrs: noinline nounwind optnone
define x86_thiscallcc void @"\01?thiscall1@Foo@@QAEXH@Z"(%struct.Foo* %this, i32 %a) #0 align 2 !dbg !111 {
entry:
  %a.addr = alloca i32, align 4
  %this.addr = alloca %struct.Foo*, align 4
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !124, metadata !DIExpression()), !dbg !125
  store %struct.Foo* %this, %struct.Foo** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo** %this.addr, metadata !126, metadata !DIExpression()), !dbg !128
  %this1 = load %struct.Foo*, %struct.Foo** %this.addr, align 4
  %0 = load i32, i32* %a.addr, align 4, !dbg !129
  %1 = load i32, i32* @g, align 4, !dbg !130
  %add = add nsw i32 %1, %0, !dbg !130
  store i32 %add, i32* @g, align 4, !dbg !130
  ret void, !dbg !131
}

; Function Attrs: noinline nounwind optnone
define x86_thiscallcc void @"\01?thiscall2@Foo@@QAEXHH@Z"(%struct.Foo* %this, i32 %a, i32 %b) #0 align 2 !dbg !132 {
entry:
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %this.addr = alloca %struct.Foo*, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !133, metadata !DIExpression()), !dbg !134
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !135, metadata !DIExpression()), !dbg !136
  store %struct.Foo* %this, %struct.Foo** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo** %this.addr, metadata !137, metadata !DIExpression()), !dbg !138
  %this1 = load %struct.Foo*, %struct.Foo** %this.addr, align 4
  %0 = load i32, i32* %a.addr, align 4, !dbg !139
  %1 = load i32, i32* %b.addr, align 4, !dbg !140
  %add = add nsw i32 %0, %1, !dbg !141
  %2 = load i32, i32* @g, align 4, !dbg !142
  %add2 = add nsw i32 %2, %add, !dbg !142
  store i32 %add2, i32* @g, align 4, !dbg !142
  ret void, !dbg !143
}

; Function Attrs: noinline nounwind optnone
define x86_thiscallcc void @"\01?thiscall3@Foo@@QAEXHHH@Z"(%struct.Foo* %this, i32 %a, i32 %b, i32 %c) #0 align 2 !dbg !144 {
entry:
  %c.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %a.addr = alloca i32, align 4
  %this.addr = alloca %struct.Foo*, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %c.addr, metadata !145, metadata !DIExpression()), !dbg !146
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %b.addr, metadata !147, metadata !DIExpression()), !dbg !148
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !149, metadata !DIExpression()), !dbg !150
  store %struct.Foo* %this, %struct.Foo** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo** %this.addr, metadata !151, metadata !DIExpression()), !dbg !152
  %this1 = load %struct.Foo*, %struct.Foo** %this.addr, align 4
  %0 = load i32, i32* %a.addr, align 4, !dbg !153
  %1 = load i32, i32* %b.addr, align 4, !dbg !154
  %add = add nsw i32 %0, %1, !dbg !155
  %2 = load i32, i32* %c.addr, align 4, !dbg !156
  %add2 = add nsw i32 %add, %2, !dbg !157
  %3 = load i32, i32* @g, align 4, !dbg !158
  %add3 = add nsw i32 %3, %add2, !dbg !158
  store i32 %add3, i32* @g, align 4, !dbg !158
  ret void, !dbg !159
}

attributes #0 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "C:\5Csrc\5Cllvm-project\5Cbuild", checksumkind: CSK_MD5, checksum: "0ce3e4edcf2f8511157da4edb99fcdf4")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "cdecl1", scope: !1, file: !1, line: 4, type: !9, isLocal: false, isDefinition: true, scopeLine: 4, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{null, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", arg: 1, scope: !8, file: !1, line: 4, type: !11)
!13 = !DILocation(line: 4, column: 17, scope: !8)
!14 = !DILocation(line: 4, column: 27, scope: !8)
!15 = !DILocation(line: 4, column: 24, scope: !8)
!16 = !DILocation(line: 4, column: 30, scope: !8)
!17 = distinct !DISubprogram(name: "cdecl2", scope: !1, file: !1, line: 5, type: !18, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !11, !11}
!20 = !DILocalVariable(name: "b", arg: 2, scope: !17, file: !1, line: 5, type: !11)
!21 = !DILocation(line: 5, column: 24, scope: !17)
!22 = !DILocalVariable(name: "a", arg: 1, scope: !17, file: !1, line: 5, type: !11)
!23 = !DILocation(line: 5, column: 17, scope: !17)
!24 = !DILocation(line: 5, column: 34, scope: !17)
!25 = !DILocation(line: 5, column: 38, scope: !17)
!26 = !DILocation(line: 5, column: 36, scope: !17)
!27 = !DILocation(line: 5, column: 31, scope: !17)
!28 = !DILocation(line: 5, column: 41, scope: !17)
!29 = distinct !DISubprogram(name: "cdecl3", scope: !1, file: !1, line: 6, type: !30, isLocal: false, isDefinition: true, scopeLine: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!30 = !DISubroutineType(types: !31)
!31 = !{null, !11, !11, !11}
!32 = !DILocalVariable(name: "c", arg: 3, scope: !29, file: !1, line: 6, type: !11)
!33 = !DILocation(line: 6, column: 31, scope: !29)
!34 = !DILocalVariable(name: "b", arg: 2, scope: !29, file: !1, line: 6, type: !11)
!35 = !DILocation(line: 6, column: 24, scope: !29)
!36 = !DILocalVariable(name: "a", arg: 1, scope: !29, file: !1, line: 6, type: !11)
!37 = !DILocation(line: 6, column: 17, scope: !29)
!38 = !DILocation(line: 6, column: 41, scope: !29)
!39 = !DILocation(line: 6, column: 45, scope: !29)
!40 = !DILocation(line: 6, column: 43, scope: !29)
!41 = !DILocation(line: 6, column: 49, scope: !29)
!42 = !DILocation(line: 6, column: 47, scope: !29)
!43 = !DILocation(line: 6, column: 38, scope: !29)
!44 = !DILocation(line: 6, column: 52, scope: !29)
!45 = distinct !DISubprogram(name: "fastcall1", linkageName: "\01@fastcall1@4", scope: !1, file: !1, line: 8, type: !46, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!46 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !10)
!47 = !DILocalVariable(name: "a", arg: 1, scope: !45, file: !1, line: 8, type: !11)
!48 = !DILocation(line: 8, column: 31, scope: !45)
!49 = !DILocation(line: 8, column: 41, scope: !45)
!50 = !DILocation(line: 8, column: 38, scope: !45)
!51 = !DILocation(line: 8, column: 44, scope: !45)
!52 = distinct !DISubprogram(name: "fastcall2", linkageName: "\01@fastcall2@8", scope: !1, file: !1, line: 9, type: !53, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!53 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !19)
!54 = !DILocalVariable(name: "b", arg: 2, scope: !52, file: !1, line: 9, type: !11)
!55 = !DILocation(line: 9, column: 38, scope: !52)
!56 = !DILocalVariable(name: "a", arg: 1, scope: !52, file: !1, line: 9, type: !11)
!57 = !DILocation(line: 9, column: 31, scope: !52)
!58 = !DILocation(line: 9, column: 48, scope: !52)
!59 = !DILocation(line: 9, column: 52, scope: !52)
!60 = !DILocation(line: 9, column: 50, scope: !52)
!61 = !DILocation(line: 9, column: 45, scope: !52)
!62 = !DILocation(line: 9, column: 55, scope: !52)
!63 = distinct !DISubprogram(name: "fastcall3", linkageName: "\01@fastcall3@12", scope: !1, file: !1, line: 10, type: !64, isLocal: false, isDefinition: true, scopeLine: 10, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!64 = !DISubroutineType(cc: DW_CC_BORLAND_msfastcall, types: !31)
!65 = !DILocalVariable(name: "c", arg: 3, scope: !63, file: !1, line: 10, type: !11)
!66 = !DILocation(line: 10, column: 45, scope: !63)
!67 = !DILocalVariable(name: "b", arg: 2, scope: !63, file: !1, line: 10, type: !11)
!68 = !DILocation(line: 10, column: 38, scope: !63)
!69 = !DILocalVariable(name: "a", arg: 1, scope: !63, file: !1, line: 10, type: !11)
!70 = !DILocation(line: 10, column: 31, scope: !63)
!71 = !DILocation(line: 10, column: 55, scope: !63)
!72 = !DILocation(line: 10, column: 59, scope: !63)
!73 = !DILocation(line: 10, column: 57, scope: !63)
!74 = !DILocation(line: 10, column: 63, scope: !63)
!75 = !DILocation(line: 10, column: 61, scope: !63)
!76 = !DILocation(line: 10, column: 52, scope: !63)
!77 = !DILocation(line: 10, column: 66, scope: !63)
!78 = distinct !DISubprogram(name: "stdcall1", linkageName: "\01_stdcall1@4", scope: !1, file: !1, line: 12, type: !79, isLocal: false, isDefinition: true, scopeLine: 12, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!79 = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: !10)
!80 = !DILocalVariable(name: "a", arg: 1, scope: !78, file: !1, line: 12, type: !11)
!81 = !DILocation(line: 12, column: 29, scope: !78)
!82 = !DILocation(line: 12, column: 39, scope: !78)
!83 = !DILocation(line: 12, column: 36, scope: !78)
!84 = !DILocation(line: 12, column: 42, scope: !78)
!85 = distinct !DISubprogram(name: "stdcall2", linkageName: "\01_stdcall2@8", scope: !1, file: !1, line: 13, type: !86, isLocal: false, isDefinition: true, scopeLine: 13, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!86 = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: !19)
!87 = !DILocalVariable(name: "b", arg: 2, scope: !85, file: !1, line: 13, type: !11)
!88 = !DILocation(line: 13, column: 36, scope: !85)
!89 = !DILocalVariable(name: "a", arg: 1, scope: !85, file: !1, line: 13, type: !11)
!90 = !DILocation(line: 13, column: 29, scope: !85)
!91 = !DILocation(line: 13, column: 46, scope: !85)
!92 = !DILocation(line: 13, column: 50, scope: !85)
!93 = !DILocation(line: 13, column: 48, scope: !85)
!94 = !DILocation(line: 13, column: 43, scope: !85)
!95 = !DILocation(line: 13, column: 53, scope: !85)
!96 = distinct !DISubprogram(name: "stdcall3", linkageName: "\01_stdcall3@12", scope: !1, file: !1, line: 14, type: !97, isLocal: false, isDefinition: true, scopeLine: 14, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!97 = !DISubroutineType(cc: DW_CC_BORLAND_stdcall, types: !31)
!98 = !DILocalVariable(name: "c", arg: 3, scope: !96, file: !1, line: 14, type: !11)
!99 = !DILocation(line: 14, column: 43, scope: !96)
!100 = !DILocalVariable(name: "b", arg: 2, scope: !96, file: !1, line: 14, type: !11)
!101 = !DILocation(line: 14, column: 36, scope: !96)
!102 = !DILocalVariable(name: "a", arg: 1, scope: !96, file: !1, line: 14, type: !11)
!103 = !DILocation(line: 14, column: 29, scope: !96)
!104 = !DILocation(line: 14, column: 53, scope: !96)
!105 = !DILocation(line: 14, column: 57, scope: !96)
!106 = !DILocation(line: 14, column: 55, scope: !96)
!107 = !DILocation(line: 14, column: 61, scope: !96)
!108 = !DILocation(line: 14, column: 59, scope: !96)
!109 = !DILocation(line: 14, column: 50, scope: !96)
!110 = !DILocation(line: 14, column: 64, scope: !96)
!111 = distinct !DISubprogram(name: "thiscall1", linkageName: "\01?thiscall1@Foo@@QAEXH@Z", scope: !112, file: !1, line: 23, type: !115, isLocal: false, isDefinition: true, scopeLine: 23, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !114, retainedNodes: !2)
!112 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !1, line: 17, size: 8, elements: !113, identifier: ".?AUFoo@@")
!113 = !{!114, !118, !121}
!114 = !DISubprogram(name: "thiscall1", linkageName: "\01?thiscall1@Foo@@QAEXH@Z", scope: !112, file: !1, line: 18, type: !115, isLocal: false, isDefinition: false, scopeLine: 18, flags: DIFlagPrototyped, isOptimized: false)
!115 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !116)
!116 = !{null, !117, !11}
!117 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !112, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!118 = !DISubprogram(name: "thiscall2", linkageName: "\01?thiscall2@Foo@@QAEXHH@Z", scope: !112, file: !1, line: 19, type: !119, isLocal: false, isDefinition: false, scopeLine: 19, flags: DIFlagPrototyped, isOptimized: false)
!119 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !120)
!120 = !{null, !117, !11, !11}
!121 = !DISubprogram(name: "thiscall3", linkageName: "\01?thiscall3@Foo@@QAEXHHH@Z", scope: !112, file: !1, line: 20, type: !122, isLocal: false, isDefinition: false, scopeLine: 20, flags: DIFlagPrototyped, isOptimized: false)
!122 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !123)
!123 = !{null, !117, !11, !11, !11}
!124 = !DILocalVariable(name: "a", arg: 2, scope: !111, file: !1, line: 23, type: !11)
!125 = !DILocation(line: 23, column: 25, scope: !111)
!126 = !DILocalVariable(name: "this", arg: 1, scope: !111, type: !127, flags: DIFlagArtificial | DIFlagObjectPointer)
!127 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !112, size: 32)
!128 = !DILocation(line: 0, scope: !111)
!129 = !DILocation(line: 23, column: 35, scope: !111)
!130 = !DILocation(line: 23, column: 32, scope: !111)
!131 = !DILocation(line: 23, column: 38, scope: !111)
!132 = distinct !DISubprogram(name: "thiscall2", linkageName: "\01?thiscall2@Foo@@QAEXHH@Z", scope: !112, file: !1, line: 24, type: !119, isLocal: false, isDefinition: true, scopeLine: 24, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !118, retainedNodes: !2)
!133 = !DILocalVariable(name: "b", arg: 3, scope: !132, file: !1, line: 24, type: !11)
!134 = !DILocation(line: 24, column: 32, scope: !132)
!135 = !DILocalVariable(name: "a", arg: 2, scope: !132, file: !1, line: 24, type: !11)
!136 = !DILocation(line: 24, column: 25, scope: !132)
!137 = !DILocalVariable(name: "this", arg: 1, scope: !132, type: !127, flags: DIFlagArtificial | DIFlagObjectPointer)
!138 = !DILocation(line: 0, scope: !132)
!139 = !DILocation(line: 24, column: 42, scope: !132)
!140 = !DILocation(line: 24, column: 46, scope: !132)
!141 = !DILocation(line: 24, column: 44, scope: !132)
!142 = !DILocation(line: 24, column: 39, scope: !132)
!143 = !DILocation(line: 24, column: 49, scope: !132)
!144 = distinct !DISubprogram(name: "thiscall3", linkageName: "\01?thiscall3@Foo@@QAEXHHH@Z", scope: !112, file: !1, line: 25, type: !122, isLocal: false, isDefinition: true, scopeLine: 25, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !121, retainedNodes: !2)
!145 = !DILocalVariable(name: "c", arg: 4, scope: !144, file: !1, line: 25, type: !11)
!146 = !DILocation(line: 25, column: 39, scope: !144)
!147 = !DILocalVariable(name: "b", arg: 3, scope: !144, file: !1, line: 25, type: !11)
!148 = !DILocation(line: 25, column: 32, scope: !144)
!149 = !DILocalVariable(name: "a", arg: 2, scope: !144, file: !1, line: 25, type: !11)
!150 = !DILocation(line: 25, column: 25, scope: !144)
!151 = !DILocalVariable(name: "this", arg: 1, scope: !144, type: !127, flags: DIFlagArtificial | DIFlagObjectPointer)
!152 = !DILocation(line: 0, scope: !144)
!153 = !DILocation(line: 25, column: 49, scope: !144)
!154 = !DILocation(line: 25, column: 53, scope: !144)
!155 = !DILocation(line: 25, column: 51, scope: !144)
!156 = !DILocation(line: 25, column: 57, scope: !144)
!157 = !DILocation(line: 25, column: 55, scope: !144)
!158 = !DILocation(line: 25, column: 46, scope: !144)
!159 = !DILocation(line: 25, column: 60, scope: !144)

; RUN: llc < %s -filetype=obj | llvm-readobj - -codeview | FileCheck %s
; RUN: llc < %s | FileCheck %s --check-prefix=ASM
; RUN: opt -S -debugger-tune=lldb %s | FileCheck -check-prefix=OPT %s
;
; -- "thunk.cpp" begin --------------------------------------------------------
; class A { public: virtual bool MyMethod() { return true; } }; 
; class B { public: virtual bool MyMethod() { return true; } }; 
; class C : public virtual A, public virtual B { 
; public: 
;     virtual bool MyMethod() { return true; } 
; }; 
; 
; int main() 
; { 
;     A* a = new C(); 
;     B* b = new C(); 
;     C* c = new C(); 
;     a->MyMethod();
;     b->MyMethod();
;     c->MyMethod();
;     bool (A::*mp)() = &A::MyMethod;
;     return 0;
; } 
; -- "thunk.cpp" end ----------------------------------------------------------
;
; Build command:
;   $ clang -S -emit-llvm -g -gcodeview thunk.cpp
;
; CHECK:       Thunk32Sym {
; CHECK-NEXT:    Kind: S_THUNK32 ({{.*}})
; CHECK-NEXT:    Name: {{.*_9A.*}}
; CHECK-NEXT:    Parent: 0
; CHECK-NEXT:    End: 0
; CHECK-NEXT:    Next: 0
; CHECK-NEXT:    Off: 0
; CHECK-NEXT:    Seg: 0
; CHECK-NEXT:    Len: {{[0-9]+}}
; CHECK-NEXT:    Ordinal: Standard (0x0)
; CHECK-NEXT:  }
; CHECK-NEXT:  ProcEnd {
; CHECK-NEXT:    Kind: S_PROC_ID_END ({{.*}})
; CHECK-NEXT:  }
;
; CHECK:       Thunk32Sym {
; CHECK-NEXT:    Kind: S_THUNK32 ({{.*}})
; CHECK-NEXT:    Name: {{.*MyMethod.*C.*}}
; CHECK-NEXT:    Parent: 0
; CHECK-NEXT:    End: 0
; CHECK-NEXT:    Next: 0
; CHECK-NEXT:    Off: 0
; CHECK-NEXT:    Seg: 0
; CHECK-NEXT:    Len: {{[0-9]+}}
; CHECK-NEXT:    Ordinal: Standard (0x0)
; CHECK-NEXT:  }
; CHECK-NEXT:  ProcEnd {
; CHECK-NEXT:    Kind: S_PROC_ID_END ({{.*}})
; CHECK-NEXT:  }

; ASM:        .long   241                     # Symbol subsection for [[NAME1:.*_9A.*]]
; ASM-NEXT:   .long   {{.*}}                  # Subsection size 
; ASM-NEXT: {{L.*}}:
; ASM-NEXT:   .short  [[END1:.?L.*]]-[[BEGIN1:.?L.*]]   # Record length 
; ASM-NEXT: [[BEGIN1]]:
; ASM-NEXT:   .short  4354                    # Record kind: S_THUNK32 
; ASM-NEXT:   .long   0                       # PtrParent 
; ASM-NEXT:   .long   0                       # PtrEnd 
; ASM-NEXT:   .long   0                       # PtrNext 
; ASM-NEXT:   .secrel32 "[[NAME1]]"           # Thunk section relative address 
; ASM-NEXT:   .secidx "[[NAME1]]"             # Thunk section index 
; ASM-NEXT:   .short  Lfunc_end{{.*}}-"[[NAME1]]" # Code size 
; ASM-NEXT:   .byte   0                       # Ordinal 
; ASM-NEXT:   .asciz  "[[NAME1]]"             # Function name 
; ASM-NEXT: [[END1]]:
; ASM-NEXT:   .short  2                       # Record length 
; ASM-NEXT:   .short  4431                    # Record kind: S_PROC_ID_END 
;
; ASM:        .long 241                       # Symbol subsection for [[NAME2:.*MyMethod.*C.*]]
; ASM-NEXT:   .long {{.*}}                    # Subsection size
; ASM-NEXT: {{L.*}}:
; ASM-NEXT:   .short [[END2:.?L.*]]-[[BEGIN2:.?L.*]] # Record length
; ASM-NEXT: [[BEGIN2]]:
; ASM-NEXT:   .short 4354                     # Record kind: S_THUNK32
; ASM-NEXT:   .long 0                         # PtrParent
; ASM-NEXT:   .long 0                         # PtrEnd
; ASM-NEXT:   .long 0                         # PtrNext
; ASM-NEXT:   .secrel32 "[[NAME2]]"           # Thunk section relative address
; ASM-NEXT:   .secidx   "[[NAME2]]"           # Thunk section index
; ASM-NEXT:   .short Lfunc_end{{.*}}-"[[NAME2]]" # Code size
; ASM-NEXT:   .byte 0                         # Ordinal
; ASM-NEXT:   .asciz "[[NAME2]]"              # Function name
; ASM-NEXT: [[END2]]:
; ASM-NEXT:   .short 2                        # Record length
; ASM-NEXT:   .short 4431                     # Record kind: S_PROC_ID_END 

; OPT: DISubprogram(linkageName: "{{.*MyMethod.*C.*}}",{{.*}} line: 5,{{.*}} flags: DIFlagArtificial | DIFlagThunk{{.*}})


; ModuleID = 'thunk.cpp'
source_filename = "thunk.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.0.24210"

%rtti.CompleteObjectLocator = type { i32, i32, i32, i8*, %rtti.ClassHierarchyDescriptor* }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, %rtti.BaseClassDescriptor** }
%rtti.BaseClassDescriptor = type { i8*, i32, i32, i32, i32, i32, %rtti.ClassHierarchyDescriptor* }
%rtti.TypeDescriptor7 = type { i8**, i8*, [8 x i8] }
%class.A = type { i32 (...)** }
%class.B = type { i32 (...)** }
%class.C = type { i32*, %class.A, %class.B }

$"\01??0C@@QAE@XZ" = comdat any

$"\01??_9A@@$BA@AE" = comdat any

$"\01??0A@@QAE@XZ" = comdat any

$"\01??0B@@QAE@XZ" = comdat any

$"\01?MyMethod@C@@UAE_NXZ" = comdat any

$"\01?MyMethod@C@@W3AE_NXZ" = comdat any

$"\01?MyMethod@A@@UAE_NXZ" = comdat any

$"\01?MyMethod@B@@UAE_NXZ" = comdat any

$"\01??_8C@@7B@" = comdat any

$"\01??_7C@@6BA@@@" = comdat largest

$"\01??_7C@@6BB@@@" = comdat largest

$"\01??_R4C@@6BA@@@" = comdat any

$"\01??_R0?AVC@@@8" = comdat any

$"\01??_R3C@@8" = comdat any

$"\01??_R2C@@8" = comdat any

$"\01??_R1A@?0A@EA@C@@8" = comdat any

$"\01??_R1A@A@3FA@A@@8" = comdat any

$"\01??_R0?AVA@@@8" = comdat any

$"\01??_R3A@@8" = comdat any

$"\01??_R2A@@8" = comdat any

$"\01??_R1A@?0A@EA@A@@8" = comdat any

$"\01??_R1A@A@7FA@B@@8" = comdat any

$"\01??_R0?AVB@@@8" = comdat any

$"\01??_R3B@@8" = comdat any

$"\01??_R2B@@8" = comdat any

$"\01??_R1A@?0A@EA@B@@8" = comdat any

$"\01??_R4C@@6BB@@@" = comdat any

$"\01??_7A@@6B@" = comdat largest

$"\01??_R4A@@6B@" = comdat any

$"\01??_7B@@6B@" = comdat largest

$"\01??_R4B@@6B@" = comdat any

@"\01??_8C@@7B@" = linkonce_odr unnamed_addr constant [3 x i32] [i32 0, i32 4, i32 8], comdat
@0 = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4C@@6BA@@@" to i8*), i8* bitcast (i1 (i8*)* @"\01?MyMethod@C@@UAE_NXZ" to i8*)] }, comdat($"\01??_7C@@6BA@@@")
@1 = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4C@@6BB@@@" to i8*), i8* bitcast (i1 (i8*)* @"\01?MyMethod@C@@W3AE_NXZ" to i8*)] }, comdat($"\01??_7C@@6BB@@@")
@"\01??_R4C@@6BA@@@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 0, i32 4, i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVC@@@8" to i8*), %rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" }, comdat
@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0?AVC@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVC@@\00" }, comdat
@"\01??_R3C@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 3, i32 3, %rtti.BaseClassDescriptor** getelementptr inbounds ([4 x %rtti.BaseClassDescriptor*], [4 x %rtti.BaseClassDescriptor*]* @"\01??_R2C@@8", i32 0, i32 0) }, comdat
@"\01??_R2C@@8" = linkonce_odr constant [4 x %rtti.BaseClassDescriptor*] [%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@C@@8", %rtti.BaseClassDescriptor* @"\01??_R1A@A@3FA@A@@8", %rtti.BaseClassDescriptor* @"\01??_R1A@A@7FA@B@@8", %rtti.BaseClassDescriptor* null], comdat
@"\01??_R1A@?0A@EA@C@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVC@@@8" to i8*), i32 2, i32 0, i32 -1, i32 0, i32 64, %rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" }, comdat
@"\01??_R1A@A@3FA@A@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVA@@@8" to i8*), i32 0, i32 0, i32 0, i32 4, i32 80, %rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" }, comdat
@"\01??_R0?AVA@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVA@@\00" }, comdat
@"\01??_R3A@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, %rtti.BaseClassDescriptor** getelementptr inbounds ([2 x %rtti.BaseClassDescriptor*], [2 x %rtti.BaseClassDescriptor*]* @"\01??_R2A@@8", i32 0, i32 0) }, comdat
@"\01??_R2A@@8" = linkonce_odr constant [2 x %rtti.BaseClassDescriptor*] [%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@A@@8", %rtti.BaseClassDescriptor* null], comdat
@"\01??_R1A@?0A@EA@A@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVA@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" }, comdat
@"\01??_R1A@A@7FA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVB@@@8" to i8*), i32 0, i32 0, i32 0, i32 8, i32 80, %rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" }, comdat
@"\01??_R0?AVB@@@8" = linkonce_odr global %rtti.TypeDescriptor7 { i8** @"\01??_7type_info@@6B@", i8* null, [8 x i8] c".?AVB@@\00" }, comdat
@"\01??_R3B@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, %rtti.BaseClassDescriptor** getelementptr inbounds ([2 x %rtti.BaseClassDescriptor*], [2 x %rtti.BaseClassDescriptor*]* @"\01??_R2B@@8", i32 0, i32 0) }, comdat
@"\01??_R2B@@8" = linkonce_odr constant [2 x %rtti.BaseClassDescriptor*] [%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@B@@8", %rtti.BaseClassDescriptor* null], comdat
@"\01??_R1A@?0A@EA@B@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVB@@@8" to i8*), i32 0, i32 0, i32 -1, i32 0, i32 64, %rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" }, comdat
@"\01??_R4C@@6BB@@@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 0, i32 8, i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVC@@@8" to i8*), %rtti.ClassHierarchyDescriptor* @"\01??_R3C@@8" }, comdat
@2 = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4A@@6B@" to i8*), i8* bitcast (i1 (%class.A*)* @"\01?MyMethod@A@@UAE_NXZ" to i8*)] }, comdat($"\01??_7A@@6B@")
@"\01??_R4A@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVA@@@8" to i8*), %rtti.ClassHierarchyDescriptor* @"\01??_R3A@@8" }, comdat
@3 = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4B@@6B@" to i8*), i8* bitcast (i1 (%class.B*)* @"\01?MyMethod@B@@UAE_NXZ" to i8*)] }, comdat($"\01??_7B@@6B@")
@"\01??_R4B@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 0, i32 0, i32 0, i8* bitcast (%rtti.TypeDescriptor7* @"\01??_R0?AVB@@@8" to i8*), %rtti.ClassHierarchyDescriptor* @"\01??_R3B@@8" }, comdat

@"\01??_7C@@6BA@@@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @0, i32 0, i32 0, i32 1)
@"\01??_7C@@6BB@@@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @1, i32 0, i32 0, i32 1)
@"\01??_7A@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @2, i32 0, i32 0, i32 1)
@"\01??_7B@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @3, i32 0, i32 0, i32 1)

; Function Attrs: noinline norecurse optnone
define i32 @main() #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %a = alloca %class.A*, align 4
  %b = alloca %class.B*, align 4
  %c = alloca %class.C*, align 4
  %mp = alloca i8*, align 4
  store i32 0, i32* %retval, align 4
  call void @llvm.dbg.declare(metadata %class.A** %a, metadata !12, metadata !DIExpression()), !dbg !24
  %call = call i8* @"\01??2@YAPAXI@Z"(i32 12) #7, !dbg !25
  %0 = bitcast i8* %call to %class.C*, !dbg !25
  %1 = bitcast %class.C* %0 to i8*, !dbg !26
  call void @llvm.memset.p0i8.i32(i8* align 8 %1, i8 0, i32 12, i1 false), !dbg !26
  %call1 = call x86_thiscallcc %class.C* @"\01??0C@@QAE@XZ"(%class.C* %0, i32 1) #8, !dbg !26
  %2 = icmp eq %class.C* %0, null, !dbg !25
  br i1 %2, label %cast.end, label %cast.notnull, !dbg !25

cast.notnull:                                     ; preds = %entry
  %3 = bitcast %class.C* %0 to i8*, !dbg !25
  %vbptr = getelementptr inbounds i8, i8* %3, i32 0, !dbg !25
  %4 = bitcast i8* %vbptr to i32**, !dbg !25
  %vbtable = load i32*, i32** %4, align 4, !dbg !25
  %5 = getelementptr inbounds i32, i32* %vbtable, i32 1, !dbg !25
  %vbase_offs = load i32, i32* %5, align 4, !dbg !25
  %6 = add nsw i32 0, %vbase_offs, !dbg !25
  %7 = bitcast %class.C* %0 to i8*, !dbg !25
  %add.ptr = getelementptr inbounds i8, i8* %7, i32 %6, !dbg !25
  %8 = bitcast i8* %add.ptr to %class.A*, !dbg !25
  br label %cast.end, !dbg !25

cast.end:                                         ; preds = %cast.notnull, %entry
  %cast.result = phi %class.A* [ %8, %cast.notnull ], [ null, %entry ], !dbg !25
  store %class.A* %cast.result, %class.A** %a, align 4, !dbg !24
  call void @llvm.dbg.declare(metadata %class.B** %b, metadata !27, metadata !DIExpression()), !dbg !36
  %call2 = call i8* @"\01??2@YAPAXI@Z"(i32 12) #7, !dbg !37
  %9 = bitcast i8* %call2 to %class.C*, !dbg !37
  %10 = bitcast %class.C* %9 to i8*, !dbg !38
  call void @llvm.memset.p0i8.i32(i8* align 8 %10, i8 0, i32 12, i1 false), !dbg !38
  %call3 = call x86_thiscallcc %class.C* @"\01??0C@@QAE@XZ"(%class.C* %9, i32 1) #8, !dbg !38
  %11 = icmp eq %class.C* %9, null, !dbg !37
  br i1 %11, label %cast.end9, label %cast.notnull4, !dbg !37

cast.notnull4:                                    ; preds = %cast.end
  %12 = bitcast %class.C* %9 to i8*, !dbg !37
  %vbptr5 = getelementptr inbounds i8, i8* %12, i32 0, !dbg !37
  %13 = bitcast i8* %vbptr5 to i32**, !dbg !37
  %vbtable6 = load i32*, i32** %13, align 4, !dbg !37
  %14 = getelementptr inbounds i32, i32* %vbtable6, i32 2, !dbg !37
  %vbase_offs7 = load i32, i32* %14, align 4, !dbg !37
  %15 = add nsw i32 0, %vbase_offs7, !dbg !37
  %16 = bitcast %class.C* %9 to i8*, !dbg !37
  %add.ptr8 = getelementptr inbounds i8, i8* %16, i32 %15, !dbg !37
  %17 = bitcast i8* %add.ptr8 to %class.B*, !dbg !37
  br label %cast.end9, !dbg !37

cast.end9:                                        ; preds = %cast.notnull4, %cast.end
  %cast.result10 = phi %class.B* [ %17, %cast.notnull4 ], [ null, %cast.end ], !dbg !37
  store %class.B* %cast.result10, %class.B** %b, align 4, !dbg !36
  call void @llvm.dbg.declare(metadata %class.C** %c, metadata !39, metadata !DIExpression()), !dbg !49
  %call11 = call i8* @"\01??2@YAPAXI@Z"(i32 12) #7, !dbg !50
  %18 = bitcast i8* %call11 to %class.C*, !dbg !50
  %19 = bitcast %class.C* %18 to i8*, !dbg !51
  call void @llvm.memset.p0i8.i32(i8* align 8 %19, i8 0, i32 12, i1 false), !dbg !51
  %call12 = call x86_thiscallcc %class.C* @"\01??0C@@QAE@XZ"(%class.C* %18, i32 1) #8, !dbg !51
  store %class.C* %18, %class.C** %c, align 4, !dbg !49
  %20 = load %class.A*, %class.A** %a, align 4, !dbg !52
  %21 = bitcast %class.A* %20 to i1 (%class.A*)***, !dbg !53
  %vtable = load i1 (%class.A*)**, i1 (%class.A*)*** %21, align 4, !dbg !53
  %vfn = getelementptr inbounds i1 (%class.A*)*, i1 (%class.A*)** %vtable, i64 0, !dbg !53
  %22 = load i1 (%class.A*)*, i1 (%class.A*)** %vfn, align 4, !dbg !53
  %call13 = call x86_thiscallcc zeroext i1 %22(%class.A* %20), !dbg !53
  %23 = load %class.B*, %class.B** %b, align 4, !dbg !54
  %24 = bitcast %class.B* %23 to i1 (%class.B*)***, !dbg !55
  %vtable14 = load i1 (%class.B*)**, i1 (%class.B*)*** %24, align 4, !dbg !55
  %vfn15 = getelementptr inbounds i1 (%class.B*)*, i1 (%class.B*)** %vtable14, i64 0, !dbg !55
  %25 = load i1 (%class.B*)*, i1 (%class.B*)** %vfn15, align 4, !dbg !55
  %call16 = call x86_thiscallcc zeroext i1 %25(%class.B* %23), !dbg !55
  %26 = load %class.C*, %class.C** %c, align 4, !dbg !56
  %27 = bitcast %class.C* %26 to i8*, !dbg !57
  %vbptr17 = getelementptr inbounds i8, i8* %27, i32 0, !dbg !57
  %28 = bitcast i8* %vbptr17 to i32**, !dbg !57
  %vbtable18 = load i32*, i32** %28, align 4, !dbg !57
  %29 = getelementptr inbounds i32, i32* %vbtable18, i32 1, !dbg !57
  %vbase_offs19 = load i32, i32* %29, align 4, !dbg !57
  %30 = add nsw i32 0, %vbase_offs19, !dbg !57
  %31 = getelementptr inbounds i8, i8* %27, i32 %30, !dbg !57
  %32 = bitcast %class.C* %26 to i8*, !dbg !57
  %vbptr20 = getelementptr inbounds i8, i8* %32, i32 0, !dbg !57
  %33 = bitcast i8* %vbptr20 to i32**, !dbg !57
  %vbtable21 = load i32*, i32** %33, align 4, !dbg !57
  %34 = getelementptr inbounds i32, i32* %vbtable21, i32 1, !dbg !57
  %vbase_offs22 = load i32, i32* %34, align 4, !dbg !57
  %35 = add nsw i32 0, %vbase_offs22, !dbg !57
  %36 = getelementptr inbounds i8, i8* %32, i32 %35, !dbg !57
  %37 = bitcast i8* %36 to i1 (i8*)***, !dbg !57
  %vtable23 = load i1 (i8*)**, i1 (i8*)*** %37, align 4, !dbg !57
  %vfn24 = getelementptr inbounds i1 (i8*)*, i1 (i8*)** %vtable23, i64 0, !dbg !57
  %38 = load i1 (i8*)*, i1 (i8*)** %vfn24, align 4, !dbg !57
  %call25 = call x86_thiscallcc zeroext i1 %38(i8* %31), !dbg !57
  call void @llvm.dbg.declare(metadata i8** %mp, metadata !58, metadata !DIExpression()), !dbg !60
  store i8* bitcast (void (%class.A*, ...)* @"\01??_9A@@$BA@AE" to i8*), i8** %mp, align 4, !dbg !60
  ret i32 0, !dbg !61
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nobuiltin
declare noalias i8* @"\01??2@YAPAXI@Z"(i32) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #3

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc %class.C* @"\01??0C@@QAE@XZ"(%class.C* returned %this, i32 %is_most_derived) unnamed_addr #4 comdat align 2 !dbg !62 {
entry:
  %retval = alloca %class.C*, align 4
  %is_most_derived.addr = alloca i32, align 4
  %this.addr = alloca %class.C*, align 4
  store i32 %is_most_derived, i32* %is_most_derived.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %is_most_derived.addr, metadata !66, metadata !DIExpression()), !dbg !67
  store %class.C* %this, %class.C** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.C** %this.addr, metadata !68, metadata !DIExpression()), !dbg !67
  %this1 = load %class.C*, %class.C** %this.addr, align 4
  store %class.C* %this1, %class.C** %retval, align 4
  %is_most_derived2 = load i32, i32* %is_most_derived.addr, align 4
  %is_complete_object = icmp ne i32 %is_most_derived2, 0, !dbg !69
  br i1 %is_complete_object, label %ctor.init_vbases, label %ctor.skip_vbases, !dbg !69

ctor.init_vbases:                                 ; preds = %entry
  %this.int8 = bitcast %class.C* %this1 to i8*, !dbg !69
  %0 = getelementptr inbounds i8, i8* %this.int8, i32 0, !dbg !69
  %vbptr.C = bitcast i8* %0 to i32**, !dbg !69
  store i32* getelementptr inbounds ([3 x i32], [3 x i32]* @"\01??_8C@@7B@", i32 0, i32 0), i32** %vbptr.C, align 4, !dbg !69
  %1 = bitcast %class.C* %this1 to i8*, !dbg !69
  %2 = getelementptr inbounds i8, i8* %1, i32 4, !dbg !69
  %3 = bitcast i8* %2 to %class.A*, !dbg !69
  %call = call x86_thiscallcc %class.A* @"\01??0A@@QAE@XZ"(%class.A* %3) #8, !dbg !69
  %4 = bitcast %class.C* %this1 to i8*, !dbg !69
  %5 = getelementptr inbounds i8, i8* %4, i32 8, !dbg !69
  %6 = bitcast i8* %5 to %class.B*, !dbg !69
  %call3 = call x86_thiscallcc %class.B* @"\01??0B@@QAE@XZ"(%class.B* %6) #8, !dbg !69
  br label %ctor.skip_vbases, !dbg !69

ctor.skip_vbases:                                 ; preds = %ctor.init_vbases, %entry
  %7 = bitcast %class.C* %this1 to i8*, !dbg !69
  %vbptr = getelementptr inbounds i8, i8* %7, i32 0, !dbg !69
  %8 = bitcast i8* %vbptr to i32**, !dbg !69
  %vbtable = load i32*, i32** %8, align 4, !dbg !69
  %9 = getelementptr inbounds i32, i32* %vbtable, i32 1, !dbg !69
  %vbase_offs = load i32, i32* %9, align 4, !dbg !69
  %10 = add nsw i32 0, %vbase_offs, !dbg !69
  %11 = bitcast %class.C* %this1 to i8*, !dbg !69
  %add.ptr = getelementptr inbounds i8, i8* %11, i32 %10, !dbg !69
  %12 = bitcast i8* %add.ptr to i32 (...)***, !dbg !69
  store i32 (...)** bitcast (i8** @"\01??_7C@@6BA@@@" to i32 (...)**), i32 (...)*** %12, align 4, !dbg !69
  %13 = bitcast %class.C* %this1 to i8*, !dbg !69
  %vbptr4 = getelementptr inbounds i8, i8* %13, i32 0, !dbg !69
  %14 = bitcast i8* %vbptr4 to i32**, !dbg !69
  %vbtable5 = load i32*, i32** %14, align 4, !dbg !69
  %15 = getelementptr inbounds i32, i32* %vbtable5, i32 2, !dbg !69
  %vbase_offs6 = load i32, i32* %15, align 4, !dbg !69
  %16 = add nsw i32 0, %vbase_offs6, !dbg !69
  %17 = bitcast %class.C* %this1 to i8*, !dbg !69
  %add.ptr7 = getelementptr inbounds i8, i8* %17, i32 %16, !dbg !69
  %18 = bitcast i8* %add.ptr7 to i32 (...)***, !dbg !69
  store i32 (...)** bitcast (i8** @"\01??_7C@@6BB@@@" to i32 (...)**), i32 (...)*** %18, align 4, !dbg !69
  %19 = load %class.C*, %class.C** %retval, align 4, !dbg !69
  ret %class.C* %19, !dbg !69
}

; Function Attrs: noinline optnone
define linkonce_odr x86_thiscallcc void @"\01??_9A@@$BA@AE"(%class.A* %this, ...) #5 comdat align 2 !dbg !70 {
entry:
  %this.addr = alloca %class.A*, align 4
  store %class.A* %this, %class.A** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !72, metadata !DIExpression()), !dbg !73
  %this1 = load %class.A*, %class.A** %this.addr, align 4
  %0 = bitcast %class.A* %this1 to void (%class.A*, ...)***
  %vtable = load void (%class.A*, ...)**, void (%class.A*, ...)*** %0, align 4
  %vfn = getelementptr inbounds void (%class.A*, ...)*, void (%class.A*, ...)** %vtable, i64 0
  %1 = load void (%class.A*, ...)*, void (%class.A*, ...)** %vfn, align 4
  musttail call x86_thiscallcc void (%class.A*, ...) %1(%class.A* %this1, ...)
  ret void
                                                  ; No predecessors!
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc %class.A* @"\01??0A@@QAE@XZ"(%class.A* returned %this) unnamed_addr #4 comdat align 2 !dbg !74 {
entry:
  %this.addr = alloca %class.A*, align 4
  store %class.A* %this, %class.A** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !78, metadata !DIExpression()), !dbg !79
  %this1 = load %class.A*, %class.A** %this.addr, align 4
  %0 = bitcast %class.A* %this1 to i32 (...)***, !dbg !80
  store i32 (...)** bitcast (i8** @"\01??_7A@@6B@" to i32 (...)**), i32 (...)*** %0, align 4, !dbg !80
  ret %class.A* %this1, !dbg !80
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc %class.B* @"\01??0B@@QAE@XZ"(%class.B* returned %this) unnamed_addr #4 comdat align 2 !dbg !81 {
entry:
  %this.addr = alloca %class.B*, align 4
  store %class.B* %this, %class.B** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.B** %this.addr, metadata !85, metadata !DIExpression()), !dbg !86
  %this1 = load %class.B*, %class.B** %this.addr, align 4
  %0 = bitcast %class.B* %this1 to i32 (...)***, !dbg !87
  store i32 (...)** bitcast (i8** @"\01??_7B@@6B@" to i32 (...)**), i32 (...)*** %0, align 4, !dbg !87
  ret %class.B* %this1, !dbg !87
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc zeroext i1 @"\01?MyMethod@C@@UAE_NXZ"(i8* %this.coerce) unnamed_addr #4 comdat align 2 !dbg !88 {
entry:
  %this = alloca %class.C*, align 4
  %this.addr = alloca %class.C*, align 4
  %coerce.val = bitcast i8* %this.coerce to %class.C*
  store %class.C* %coerce.val, %class.C** %this, align 4
  %this1 = load %class.C*, %class.C** %this, align 4
  store %class.C* %this1, %class.C** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.C** %this.addr, metadata !89, metadata !DIExpression()), !dbg !90
  %this2 = load %class.C*, %class.C** %this.addr, align 4
  %0 = bitcast %class.C* %this2 to i8*
  %1 = getelementptr inbounds i8, i8* %0, i32 -4
  %this.adjusted = bitcast i8* %1 to %class.C*
  ret i1 true, !dbg !91
}

; Function Attrs: noinline optnone
define linkonce_odr x86_thiscallcc zeroext i1 @"\01?MyMethod@C@@W3AE_NXZ"(i8* %this.coerce) unnamed_addr #6 comdat align 2 !dbg !92 {
entry:
  %this = alloca %class.C*, align 4
  %this.addr = alloca %class.C*, align 4
  %coerce.val = bitcast i8* %this.coerce to %class.C*
  store %class.C* %coerce.val, %class.C** %this, align 4
  %this1 = load %class.C*, %class.C** %this, align 4
  store %class.C* %this1, %class.C** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.C** %this.addr, metadata !93, metadata !DIExpression()), !dbg !94
  %this2 = load %class.C*, %class.C** %this.addr, align 4, !dbg !94
  %0 = bitcast %class.C* %this2 to i8*, !dbg !94
  %1 = getelementptr i8, i8* %0, i32 -4, !dbg !94
  %call = tail call x86_thiscallcc zeroext i1 @"\01?MyMethod@C@@UAE_NXZ"(i8* %1), !dbg !94
  ret i1 %call, !dbg !94
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc zeroext i1 @"\01?MyMethod@A@@UAE_NXZ"(%class.A* %this) unnamed_addr #4 comdat align 2 !dbg !95 {
entry:
  %this.addr = alloca %class.A*, align 4
  store %class.A* %this, %class.A** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.A** %this.addr, metadata !96, metadata !DIExpression()), !dbg !97
  %this1 = load %class.A*, %class.A** %this.addr, align 4
  ret i1 true, !dbg !98
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc zeroext i1 @"\01?MyMethod@B@@UAE_NXZ"(%class.B* %this) unnamed_addr #4 comdat align 2 !dbg !99 {
entry:
  %this.addr = alloca %class.B*, align 4
  store %class.B* %this, %class.B** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %class.B** %this.addr, metadata !100, metadata !DIExpression()), !dbg !101
  %this1 = load %class.B*, %class.B** %this.addr, align 4
  ret i1 true, !dbg !102
}

attributes #0 = { noinline norecurse optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { nobuiltin "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { argmemonly nounwind }
attributes #4 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #5 = { noinline optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "thunk" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #6 = { noinline optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="pentium4" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #7 = { builtin }
attributes #8 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 7.0.0 (trunk)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "thunk.cpp", directory: "C:\5Cpath\5Cto", checksumkind: CSK_MD5, checksum: "d4b977fc614313c5a4fbdc980b2f4243")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 7.0.0 (trunk)"}
!8 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 8, type: !9, isLocal: false, isDefinition: true, scopeLine: 9, flags: DIFlagPrototyped, isOptimized: false, unit: !0, variables: !2)
!9 = !DISubroutineType(types: !10)
!10 = !{!11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "a", scope: !8, file: !1, line: 10, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32)
!14 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "A", file: !1, line: 1, size: 32, flags: DIFlagTypePassByValue, elements: !15, vtableHolder: !14, identifier: ".?AVA@@")
!15 = !{!16, !17, !19}
!16 = !DIDerivedType(tag: DW_TAG_pointer_type, name: "__vtbl_ptr_type", baseType: null, size: 32)
!17 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$A", scope: !1, file: !1, baseType: !18, size: 32, flags: DIFlagArtificial)
!18 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !16, size: 32)
!19 = !DISubprogram(name: "MyMethod", linkageName: "\01?MyMethod@A@@UAE_NXZ", scope: !14, file: !1, line: 1, type: !20, isLocal: false, isDefinition: false, scopeLine: 1, containingType: !14, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPublic | DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!20 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !21)
!21 = !{!22, !23}
!22 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!24 = !DILocation(line: 10, column: 8, scope: !8)
!25 = !DILocation(line: 10, column: 12, scope: !8)
!26 = !DILocation(line: 10, column: 16, scope: !8)
!27 = !DILocalVariable(name: "b", scope: !8, file: !1, line: 11, type: !28)
!28 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 32)
!29 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "B", file: !1, line: 2, size: 32, flags: DIFlagTypePassByValue, elements: !30, vtableHolder: !29, identifier: ".?AVB@@")
!30 = !{!16, !31, !32}
!31 = !DIDerivedType(tag: DW_TAG_member, name: "_vptr$B", scope: !1, file: !1, baseType: !18, size: 32, flags: DIFlagArtificial)
!32 = !DISubprogram(name: "MyMethod", linkageName: "\01?MyMethod@B@@UAE_NXZ", scope: !29, file: !1, line: 2, type: !33, isLocal: false, isDefinition: false, scopeLine: 2, containingType: !29, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, flags: DIFlagPublic | DIFlagPrototyped | DIFlagIntroducedVirtual, isOptimized: false)
!33 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !34)
!34 = !{!22, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !29, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!36 = !DILocation(line: 11, column: 8, scope: !8)
!37 = !DILocation(line: 11, column: 12, scope: !8)
!38 = !DILocation(line: 11, column: 16, scope: !8)
!39 = !DILocalVariable(name: "c", scope: !8, file: !1, line: 12, type: !40)
!40 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 32)
!41 = distinct !DICompositeType(tag: DW_TAG_class_type, name: "C", file: !1, line: 3, size: 96, flags: DIFlagTypePassByValue, elements: !42, vtableHolder: !41, identifier: ".?AVC@@")
!42 = !{!43, !44, !45}
!43 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !41, baseType: !14, offset: 4, flags: DIFlagPublic | DIFlagVirtual)
!44 = !DIDerivedType(tag: DW_TAG_inheritance, scope: !41, baseType: !29, offset: 8, flags: DIFlagPublic | DIFlagVirtual)
!45 = !DISubprogram(name: "MyMethod", linkageName: "\01?MyMethod@C@@UAE_NXZ", scope: !41, file: !1, line: 5, type: !46, isLocal: false, isDefinition: false, scopeLine: 5, containingType: !41, virtuality: DW_VIRTUALITY_virtual, virtualIndex: 0, thisAdjustment: 4, flags: DIFlagPublic | DIFlagPrototyped, isOptimized: false)
!46 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !47)
!47 = !{!22, !48}
!48 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !41, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!49 = !DILocation(line: 12, column: 8, scope: !8)
!50 = !DILocation(line: 12, column: 12, scope: !8)
!51 = !DILocation(line: 12, column: 16, scope: !8)
!52 = !DILocation(line: 13, column: 5, scope: !8)
!53 = !DILocation(line: 13, column: 8, scope: !8)
!54 = !DILocation(line: 14, column: 5, scope: !8)
!55 = !DILocation(line: 14, column: 8, scope: !8)
!56 = !DILocation(line: 15, column: 5, scope: !8)
!57 = !DILocation(line: 15, column: 8, scope: !8)
!58 = !DILocalVariable(name: "mp", scope: !8, file: !1, line: 16, type: !59)
!59 = !DIDerivedType(tag: DW_TAG_ptr_to_member_type, baseType: !20, size: 32, flags: DIFlagSingleInheritance, extraData: !14)
!60 = !DILocation(line: 16, column: 15, scope: !8)
!61 = !DILocation(line: 17, column: 5, scope: !8)
!62 = distinct !DISubprogram(name: "C", linkageName: "\01??0C@@QAE@XZ", scope: !41, file: !1, line: 3, type: !63, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !65, variables: !2)
!63 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !64)
!64 = !{null, !48}
!65 = !DISubprogram(name: "C", scope: !41, type: !63, isLocal: false, isDefinition: false, flags: DIFlagPublic | DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!66 = !DILocalVariable(name: "is_most_derived", arg: 2, scope: !62, type: !11, flags: DIFlagArtificial)
!67 = !DILocation(line: 0, scope: !62)
!68 = !DILocalVariable(name: "this", arg: 1, scope: !62, type: !40, flags: DIFlagArtificial | DIFlagObjectPointer)
!69 = !DILocation(line: 3, column: 7, scope: !62)
!70 = distinct !DISubprogram(linkageName: "\01??_9A@@$BA@AE", scope: !1, file: !1, line: 1, type: !71, isLocal: false, isDefinition: true, flags: DIFlagArtificial | DIFlagThunk, isOptimized: false, unit: !0, variables: !2)
!71 = !DISubroutineType(types: !2)
!72 = !DILocalVariable(name: "this", arg: 1, scope: !70, type: !13, flags: DIFlagArtificial | DIFlagObjectPointer)
!73 = !DILocation(line: 0, scope: !70)
!74 = distinct !DISubprogram(name: "A", linkageName: "\01??0A@@QAE@XZ", scope: !14, file: !1, line: 1, type: !75, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !77, variables: !2)
!75 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !76)
!76 = !{null, !23}
!77 = !DISubprogram(name: "A", scope: !14, type: !75, isLocal: false, isDefinition: false, flags: DIFlagPublic | DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!78 = !DILocalVariable(name: "this", arg: 1, scope: !74, type: !13, flags: DIFlagArtificial | DIFlagObjectPointer)
!79 = !DILocation(line: 0, scope: !74)
!80 = !DILocation(line: 1, column: 7, scope: !74)
!81 = distinct !DISubprogram(name: "B", linkageName: "\01??0B@@QAE@XZ", scope: !29, file: !1, line: 2, type: !82, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagArtificial | DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !84, variables: !2)
!82 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !83)
!83 = !{null, !35}
!84 = !DISubprogram(name: "B", scope: !29, type: !82, isLocal: false, isDefinition: false, flags: DIFlagPublic | DIFlagArtificial | DIFlagPrototyped, isOptimized: false)
!85 = !DILocalVariable(name: "this", arg: 1, scope: !81, type: !28, flags: DIFlagArtificial | DIFlagObjectPointer)
!86 = !DILocation(line: 0, scope: !81)
!87 = !DILocation(line: 2, column: 7, scope: !81)
!88 = distinct !DISubprogram(name: "MyMethod", linkageName: "\01?MyMethod@C@@UAE_NXZ", scope: !41, file: !1, line: 5, type: !46, isLocal: false, isDefinition: true, scopeLine: 5, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !45, variables: !2)
!89 = !DILocalVariable(name: "this", arg: 1, scope: !88, type: !40, flags: DIFlagArtificial | DIFlagObjectPointer)
!90 = !DILocation(line: 0, scope: !88)
!91 = !DILocation(line: 5, column: 31, scope: !88)
!92 = distinct !DISubprogram(linkageName: "\01?MyMethod@C@@W3AE_NXZ", scope: !1, file: !1, line: 5, type: !71, isLocal: false, isDefinition: true, flags: DIFlagArtificial | DIFlagThunk, isOptimized: false, unit: !0, variables: !2)
!93 = !DILocalVariable(name: "this", arg: 1, scope: !92, type: !40, flags: DIFlagArtificial | DIFlagObjectPointer)
!94 = !DILocation(line: 0, scope: !92)
!95 = distinct !DISubprogram(name: "MyMethod", linkageName: "\01?MyMethod@A@@UAE_NXZ", scope: !14, file: !1, line: 1, type: !20, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !19, variables: !2)
!96 = !DILocalVariable(name: "this", arg: 1, scope: !95, type: !13, flags: DIFlagArtificial | DIFlagObjectPointer)
!97 = !DILocation(line: 0, scope: !95)
!98 = !DILocation(line: 1, column: 45, scope: !95)
!99 = distinct !DISubprogram(name: "MyMethod", linkageName: "\01?MyMethod@B@@UAE_NXZ", scope: !29, file: !1, line: 2, type: !33, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !32, variables: !2)
!100 = !DILocalVariable(name: "this", arg: 1, scope: !99, type: !28, flags: DIFlagArtificial | DIFlagObjectPointer)
!101 = !DILocation(line: 0, scope: !99)
!102 = !DILocation(line: 2, column: 45, scope: !99)

; RUN: llc < %s -mtriple=x86_64-pc-windows-msvc | FileCheck %s
; Control Flow Guard is currently only available on Windows

; CHECK: .set @feat.00, 2048

; CHECK: .section .gfids$y
; CHECK: .symidx "?address_taken@@YAXXZ"
; CHECK: .symidx "?virt_method@Derived@@UEBAHXZ"

; ModuleID = 'cfguard.cpp'
source_filename = "cfguard.cpp"
target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

%struct.Derived = type { %struct.Base }
%struct.Base = type { i32 (...)** }
%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor13 = type { i8**, i8*, [14 x i8] }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor10 = type { i8**, i8*, [11 x i8] }

$"\01??0Derived@@QEAA@XZ" = comdat any

$"\01??0Base@@QEAA@XZ" = comdat any

$"\01?virt_method@Derived@@UEBAHXZ" = comdat any

$"\01??_7Derived@@6B@" = comdat largest

$"\01??_R4Derived@@6B@" = comdat any

$"\01??_R0?AUDerived@@@8" = comdat any

$"\01??_R3Derived@@8" = comdat any

$"\01??_R2Derived@@8" = comdat any

$"\01??_R1A@?0A@EA@Derived@@8" = comdat any

$"\01??_R1A@?0A@EA@Base@@8" = comdat any

$"\01??_R0?AUBase@@@8" = comdat any

$"\01??_R3Base@@8" = comdat any

$"\01??_R2Base@@8" = comdat any

$"\01??_7Base@@6B@" = comdat largest

$"\01??_R4Base@@6B@" = comdat any

@"\01?D@@3UDerived@@A" = global %struct.Derived zeroinitializer, align 8
@0 = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4Derived@@6B@" to i8*), i8* bitcast (i32 (%struct.Derived*)* @"\01?virt_method@Derived@@UEBAHXZ" to i8*)] }, comdat($"\01??_7Derived@@6B@")
@"\01??_R4Derived@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor13* @"\01??_R0?AUDerived@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3Derived@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4Derived@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0?AUDerived@@@8" = linkonce_odr global %rtti.TypeDescriptor13 { i8** @"\01??_7type_info@@6B@", i8* null, [14 x i8] c".?AUDerived@@\00" }, comdat
@__ImageBase = external constant i8
@"\01??_R3Derived@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([3 x i32]* @"\01??_R2Derived@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2Derived@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@Derived@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@Base@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@Derived@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor13* @"\01??_R0?AUDerived@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3Derived@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R1A@?0A@EA@Base@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor10* @"\01??_R0?AUBase@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3Base@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUBase@@@8" = linkonce_odr global %rtti.TypeDescriptor10 { i8** @"\01??_7type_info@@6B@", i8* null, [11 x i8] c".?AUBase@@\00" }, comdat
@"\01??_R3Base@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint ([2 x i32]* @"\01??_R2Base@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2Base@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.BaseClassDescriptor* @"\01??_R1A@?0A@EA@Base@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 0], comdat
@1 = private unnamed_addr constant { [2 x i8*] } { [2 x i8*] [i8* bitcast (%rtti.CompleteObjectLocator* @"\01??_R4Base@@6B@" to i8*), i8* bitcast (void ()* @_purecall to i8*)] }, comdat($"\01??_7Base@@6B@")
@"\01??_R4Base@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.TypeDescriptor10* @"\01??_R0?AUBase@@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.ClassHierarchyDescriptor* @"\01??_R3Base@@8" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (%rtti.CompleteObjectLocator* @"\01??_R4Base@@6B@" to i64), i64 ptrtoint (i8* @__ImageBase to i64)) to i32) }, comdat
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_cfguard.cpp, i8* null }]

@"\01??_7Derived@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @0, i32 0, i32 0, i32 1)
@"\01??_7Base@@6B@" = unnamed_addr alias i8*, getelementptr inbounds ({ [2 x i8*] }, { [2 x i8*] }* @1, i32 0, i32 0, i32 1)

; Function Attrs: noinline nounwind
define internal void @"\01??__ED@@YAXXZ"() #0 {
entry:
  %call = call %struct.Derived* @"\01??0Derived@@QEAA@XZ"(%struct.Derived* @"\01?D@@3UDerived@@A") #2
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr %struct.Derived* @"\01??0Derived@@QEAA@XZ"(%struct.Derived* returned %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.Derived*, align 8
  store %struct.Derived* %this, %struct.Derived** %this.addr, align 8
  %this1 = load %struct.Derived*, %struct.Derived** %this.addr, align 8
  %0 = bitcast %struct.Derived* %this1 to %struct.Base*
  %call = call %struct.Base* @"\01??0Base@@QEAA@XZ"(%struct.Base* %0) #2
  %1 = bitcast %struct.Derived* %this1 to i32 (...)***
  store i32 (...)** bitcast (i8** @"\01??_7Derived@@6B@" to i32 (...)**), i32 (...)*** %1, align 8
  ret %struct.Derived* %this1
}

; Function Attrs: noinline nounwind optnone
define void @"\01?address_taken@@YAXXZ"() #1 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone
define void ()* @"\01?foo@@YAP6AXXZPEAUBase@@@Z"(%struct.Base* %B) #1 {
entry:
  %retval = alloca void ()*, align 8
  %B.addr = alloca %struct.Base*, align 8
  store %struct.Base* %B, %struct.Base** %B.addr, align 8
  %0 = load %struct.Base*, %struct.Base** %B.addr, align 8
  %1 = bitcast %struct.Base* %0 to i32 (%struct.Base*)***
  %vtable = load i32 (%struct.Base*)**, i32 (%struct.Base*)*** %1, align 8
  %vfn = getelementptr inbounds i32 (%struct.Base*)*, i32 (%struct.Base*)** %vtable, i64 0
  %2 = load i32 (%struct.Base*)*, i32 (%struct.Base*)** %vfn, align 8
  %call = call i32 %2(%struct.Base* %0)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store void ()* @"\01?address_taken@@YAXXZ", void ()** %retval, align 8
  br label %return

if.end:                                           ; preds = %entry
  store void ()* null, void ()** %retval, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %3 = load void ()*, void ()** %retval, align 8
  ret void ()* %3
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr %struct.Base* @"\01??0Base@@QEAA@XZ"(%struct.Base* returned %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.Base*, align 8
  store %struct.Base* %this, %struct.Base** %this.addr, align 8
  %this1 = load %struct.Base*, %struct.Base** %this.addr, align 8
  %0 = bitcast %struct.Base* %this1 to i32 (...)***
  store i32 (...)** bitcast (i8** @"\01??_7Base@@6B@" to i32 (...)**), i32 (...)*** %0, align 8
  ret %struct.Base* %this1
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr i32 @"\01?virt_method@Derived@@UEBAHXZ"(%struct.Derived* %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %struct.Derived*, align 8
  store %struct.Derived* %this, %struct.Derived** %this.addr, align 8
  %this1 = load %struct.Derived*, %struct.Derived** %this.addr, align 8
  ret i32 42
}

declare dllimport void @_purecall() unnamed_addr

; Function Attrs: noinline nounwind
define internal void @_GLOBAL__sub_I_cfguard.cpp() #0 {
entry:
  call void @"\01??__ED@@YAXXZ"()
  ret void
}

attributes #0 = { noinline nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 2, !"cfguard", i32 1}
!1 = !{i32 1, !"wchar_size", i32 2}
!2 = !{!"clang version 6.0.0 "}

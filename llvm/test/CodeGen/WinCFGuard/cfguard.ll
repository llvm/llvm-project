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
%struct.Base = type { ptr }
%rtti.CompleteObjectLocator = type { i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor13 = type { ptr, ptr, [14 x i8] }
%rtti.ClassHierarchyDescriptor = type { i32, i32, i32, i32 }
%rtti.BaseClassDescriptor = type { i32, i32, i32, i32, i32, i32, i32 }
%rtti.TypeDescriptor10 = type { ptr, ptr, [11 x i8] }

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
@0 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"\01??_R4Derived@@6B@", ptr @"\01?virt_method@Derived@@UEBAHXZ"] }, comdat($"\01??_7Derived@@6B@")
@"\01??_R4Derived@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUDerived@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3Derived@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R4Derived@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_7type_info@@6B@" = external constant ptr
@"\01??_R0?AUDerived@@@8" = linkonce_odr global %rtti.TypeDescriptor13 { ptr @"\01??_7type_info@@6B@", ptr null, [14 x i8] c".?AUDerived@@\00" }, comdat
@__ImageBase = external constant i8
@"\01??_R3Derived@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 2, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R2Derived@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2Derived@@8" = linkonce_odr constant [3 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@Derived@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@Base@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@"\01??_R1A@?0A@EA@Derived@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUDerived@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 1, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3Derived@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R1A@?0A@EA@Base@@8" = linkonce_odr constant %rtti.BaseClassDescriptor { i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUBase@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0, i32 0, i32 -1, i32 0, i32 64, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3Base@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R0?AUBase@@@8" = linkonce_odr global %rtti.TypeDescriptor10 { ptr @"\01??_7type_info@@6B@", ptr null, [11 x i8] c".?AUBase@@\00" }, comdat
@"\01??_R3Base@@8" = linkonce_odr constant %rtti.ClassHierarchyDescriptor { i32 0, i32 0, i32 1, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R2Base@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@"\01??_R2Base@@8" = linkonce_odr constant [2 x i32] [i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R1A@?0A@EA@Base@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 0], comdat
@1 = private unnamed_addr constant { [2 x ptr] } { [2 x ptr] [ptr @"\01??_R4Base@@6B@", ptr @_purecall] }, comdat($"\01??_7Base@@6B@")
@"\01??_R4Base@@6B@" = linkonce_odr constant %rtti.CompleteObjectLocator { i32 1, i32 0, i32 0, i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R0?AUBase@@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R3Base@@8" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32), i32 trunc (i64 sub nuw nsw (i64 ptrtoint (ptr @"\01??_R4Base@@6B@" to i64), i64 ptrtoint (ptr @__ImageBase to i64)) to i32) }, comdat
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_cfguard.cpp, ptr null }]

@"\01??_7Derived@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @0, i32 0, i32 0, i32 1)
@"\01??_7Base@@6B@" = unnamed_addr alias ptr, getelementptr inbounds ({ [2 x ptr] }, ptr @1, i32 0, i32 0, i32 1)

; Function Attrs: noinline nounwind
define internal void @"\01??__ED@@YAXXZ"() #0 {
entry:
  %call = call ptr @"\01??0Derived@@QEAA@XZ"(ptr @"\01?D@@3UDerived@@A") #2
  ret void
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr ptr @"\01??0Derived@@QEAA@XZ"(ptr returned %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = bitcast ptr %this1 to ptr
  %call = call ptr @"\01??0Base@@QEAA@XZ"(ptr %0) #2
  %1 = bitcast ptr %this1 to ptr
  store ptr @"\01??_7Derived@@6B@", ptr %1, align 8
  ret ptr %this1
}

; Function Attrs: noinline nounwind optnone
define void @"\01?address_taken@@YAXXZ"() #1 {
entry:
  ret void
}

; Function Attrs: noinline nounwind optnone
define ptr @"\01?foo@@YAP6AXXZPEAUBase@@@Z"(ptr %B) #1 {
entry:
  %retval = alloca ptr, align 8
  %B.addr = alloca ptr, align 8
  store ptr %B, ptr %B.addr, align 8
  %0 = load ptr, ptr %B.addr, align 8
  %1 = bitcast ptr %0 to ptr
  %vtable = load ptr, ptr %1, align 8
  %vfn = getelementptr inbounds ptr, ptr %vtable, i64 0
  %2 = load ptr, ptr %vfn, align 8
  %call = call i32 %2(ptr %0)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store ptr @"\01?address_taken@@YAXXZ", ptr %retval, align 8
  br label %return

if.end:                                           ; preds = %entry
  store ptr null, ptr %retval, align 8
  br label %return

return:                                           ; preds = %if.end, %if.then
  %3 = load ptr, ptr %retval, align 8
  ret ptr %3
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr ptr @"\01??0Base@@QEAA@XZ"(ptr returned %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
  %0 = bitcast ptr %this1 to ptr
  store ptr @"\01??_7Base@@6B@", ptr %0, align 8
  ret ptr %this1
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr i32 @"\01?virt_method@Derived@@UEBAHXZ"(ptr %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca ptr, align 8
  store ptr %this, ptr %this.addr, align 8
  %this1 = load ptr, ptr %this.addr, align 8
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

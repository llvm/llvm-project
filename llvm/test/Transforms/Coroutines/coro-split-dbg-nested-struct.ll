; RUN: opt < %s -passes='cgscc(coro-split)' -S | FileCheck %s

; Test that nested structs in coroutine frames have correct debug info scoping.

; Minimal nested struct types that used to trigger a scoping issue:
; we used to set the wrong `scope` for the `DIDerivedType` member entries of the `DICompositeType`
; as well as the `scope` for `DICompositeType` for the inner struct itself.
%"struct.Inner" = type { i32, ptr }
%"struct.Outer" = type { %"struct.Inner", i64 }
%"class.Promise" = type { %"struct.Outer" }

define void @test_coro_function() presplitcoroutine !dbg !10 {
entry:
  %__promise = alloca %"class.Promise", align 8
  %0 = call token @llvm.coro.id(i32 0, ptr %__promise, ptr null, ptr null)
  %1 = call ptr @llvm.coro.begin(token %0, ptr null)
  %2 = call token @llvm.coro.save(ptr null)
  ret void
}

; CHECK: define void @test_coro_function()

; Check that frame debug info is generated
; CHECK: ![[FRAME_TYPE:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "{{.*}}.coro_frame_ty"

; Key validation: Check that nested structs have the correct scope hierarchy
; 1. Promise should be scoped to the frame
; CHECK: ![[PROMISE:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "class_Promise", scope: ![[FRAME_TYPE]]

; 2. Members of Promise should be scoped to Promise (check this before Outer since it comes first in output)
; CHECK: !DIDerivedType(tag: DW_TAG_member, name: "struct_Outer", scope: ![[PROMISE]]

; 3. Outer should be scoped to Promise (not the frame!)
; CHECK: ![[OUTER:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "struct_Outer", scope: ![[PROMISE]]

; 4. First Outer member should be scoped to Outer
; CHECK: !DIDerivedType(tag: DW_TAG_member, name: "struct_Inner", scope: ![[OUTER]]

; 5. Inner should be scoped to Outer (proper nesting)
; CHECK: ![[INNER:[0-9]+]] = !DICompositeType(tag: DW_TAG_structure_type, name: "struct_Inner", scope: ![[OUTER]]

; 6. Members of Inner should be scoped to Inner
; CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__int_32", scope: ![[INNER]]
; CHECK: !DIDerivedType(tag: DW_TAG_member, name: "PointerType", scope: ![[INNER]]

; 7. Second Outer member comes after Inner (due to output order)
; CHECK: !DIDerivedType(tag: DW_TAG_member, name: "__int_64", scope: ![[OUTER]]

declare token @llvm.coro.id(i32, ptr readnone, ptr readonly, ptr)
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare token @llvm.coro.save(ptr)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: ".")
!9 = !{i32 2, !"Debug Info Version", i32 3}
!10 = distinct !DISubprogram(name: "test_coro_function", scope: !1, file: !1, line: 1, type: !11, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}

; RUN: split-file %s %t
; RUN: cat %t/shared.ll %t/unsupported.ll | opt -mtriple=bpfel -O2 -o %t/unsupported.bc
; RUN: not llc -filetype=obj %t/unsupported.bc -o %t/unsupported.o 2>&1 | FileCheck %s --check-prefix=ERROR
; RUN: cat %t/shared.ll %t/supported.ll | opt -mtriple=bpfel -O2 | llc -o - | FileCheck %s --check-prefix=SUPPORTED

; The first relocation comes from:
;   namespace arg { struct T {}; }
;   namespace A { template<class> struct S { int a, b; }; }
;   int member_offset(A::S<arg::T> *p) {
;     return __builtin_preserve_field_info(p->b, 0);
;   }
; BTF emits only S<arg::T>, which also describes B::S<arg::T>. Reject the
; relocation before decoding its legacy carrier name, even when the C++ use
; was inlined into C.
; The second relocation checks record scope, a named typedef root, the
; distinct type-ID carrier, and the separately versioned C++ language form.
; Function-local C++ types also lose their scope.
; ERROR: test.cpp:7:10: {{.*}}CO-RE relocation for 'S<arg::T>' is unsupported because its BTF type name omits the enclosing C++ scope
; ERROR: CO-RE relocation for 'Alias' is unsupported because its BTF type name omits the enclosing C++ scope
; ERROR: CO-RE relocation for 'Local' is unsupported because its BTF type name omits the enclosing C++ scope

; Local type IDs do not match target types. A global template with a qualified
; argument, a global typedef of a scoped type, and a Rust module type must not
; be rejected. All four operations must still emit their relocation records.
; SUPPORTED: .long 16 {{.*}} FieldReloc
; SUPPORTED-NEXT: .long {{[0-9]+}} {{.*}} Field reloc section
; SUPPORTED-NEXT: .long 4

;--- shared.ll
target triple = "bpf"
%struct.S = type { i32, i32 }

declare ptr @llvm.preserve.struct.access.index.p0.p0(ptr, i32 immarg, i32 immarg)
declare i32 @llvm.bpf.preserve.field.info.p0(ptr, i64 immarg)
declare i64 @llvm.bpf.btf.type.id(i32, i64)

!llvm.module.flags = !{!14}
!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!3 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<arg::T>", scope: !4, file: !1, size: 64, elements: !5, templateParams: !8)
!4 = !DINamespace(name: "A", scope: null)
!5 = !{!6, !7}
!6 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !3, file: !1, baseType: !2, size: 32)
!7 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !3, file: !1, baseType: !2, size: 32, offset: 32)
!8 = !{!DITemplateTypeParameter(type: !9)}
!9 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "T", scope: !DINamespace(name: "arg", scope: null), file: !1, size: 8, elements: !{})
!10 = !DISubroutineType(types: !{!2, !11})
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !3, size: 64)
!12 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!13 = !DISubroutineType(types: !{!12})
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "S<arg::T>", file: !1, size: 8, elements: !{}, templateParams: !8)
!17 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Outer", file: !1, size: 8, elements: !{})
!18 = !DIDerivedType(tag: DW_TAG_typedef, name: "Alias", file: !1, baseType: !3)
!19 = !DIDerivedType(tag: DW_TAG_typedef, name: "Alias", scope: !17, file: !1, baseType: !2)
!20 = distinct !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus, sourceLanguageVersion: 202002, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!21 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !22, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!22 = !DIFile(filename: "test.rs", directory: "/")
!23 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "RustRecord", scope: !DINamespace(name: "module", scope: null), file: !22, size: 8, elements: !{})

;--- unsupported.ll
define i32 @call_cpp(ptr %p) !dbg !30 {
  %field = call ptr @llvm.preserve.struct.access.index.p0.p0(ptr elementtype(%struct.S) %p, i32 1, i32 1), !llvm.preserve.access.index !3, !dbg !31
  %offset = call i32 @llvm.bpf.preserve.field.info.p0(ptr %field, i64 0), !dbg !31
  ret i32 %offset, !dbg !31
}

define i64 @remote_id() !dbg !32 {
  %id = call i64 @llvm.bpf.btf.type.id(i32 0, i64 1), !llvm.preserve.access.index !19
  ret i64 %id
}

define i64 @local_scope() !dbg !36 {
  %id = call i64 @llvm.bpf.btf.type.id(i32 1, i64 1), !llvm.preserve.access.index !37
  ret i64 %id
}

!llvm.dbg.cu = !{!0, !20, !27}
!26 = !DIFile(filename: "caller.c", directory: "/")
!27 = distinct !DICompileUnit(language: DW_LANG_C11, file: !26, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!28 = !DISubroutineType(types: !{!2, !29})
!29 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!30 = distinct !DISubprogram(name: "call_cpp", scope: !26, file: !26, type: !28, spFlags: DISPFlagDefinition, unit: !27)
!31 = !DILocation(line: 7, column: 10, scope: !34, inlinedAt: !35)
!32 = distinct !DISubprogram(name: "remote_id", scope: !1, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !20)
!34 = distinct !DISubprogram(name: "member_offset", scope: !1, file: !1, type: !10, spFlags: DISPFlagDefinition, unit: !0)
!35 = !DILocation(line: 4, column: 5, scope: !30)
!36 = distinct !DISubprogram(name: "local_scope", scope: !1, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!37 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Local", scope: !36, file: !1, size: 8, elements: !{})

;--- supported.ll
define i64 @local_id() !dbg !40 {
  %id = call i64 @llvm.bpf.btf.type.id(i32 0, i64 0), !llvm.preserve.access.index !3
  ret i64 %id
}

define i64 @global_template() !dbg !41 {
  %id = call i64 @llvm.bpf.btf.type.id(i32 1, i64 1), !llvm.preserve.access.index !15
  ret i64 %id
}

define i64 @global_alias() !dbg !42 {
  %id = call i64 @llvm.bpf.btf.type.id(i32 2, i64 1), !llvm.preserve.access.index !18
  ret i64 %id
}

define i64 @rust_module() !dbg !43 {
  %id = call i64 @llvm.bpf.btf.type.id(i32 3, i64 1), !llvm.preserve.access.index !23
  ret i64 %id
}

!llvm.dbg.cu = !{!0, !21}
!40 = distinct !DISubprogram(name: "local_id", scope: !1, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!41 = distinct !DISubprogram(name: "global_template", scope: !1, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!42 = distinct !DISubprogram(name: "global_alias", scope: !1, file: !1, type: !13, spFlags: DISPFlagDefinition, unit: !0)
!43 = distinct !DISubprogram(name: "rust_module", scope: !22, file: !22, type: !13, spFlags: DISPFlagDefinition, unit: !21)

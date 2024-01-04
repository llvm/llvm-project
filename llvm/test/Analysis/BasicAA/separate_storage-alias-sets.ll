; We want BasicAA to make a reasonable conservative guess as to context for
; separate storage hints. This lets alias analysis users (such as the alias set
; tracker) who can't be context-sensitive still get the benefits of hints.

; RUN: opt < %s -basic-aa-separate-storage -S -passes=print-alias-sets 2>&1 | FileCheck %s

declare void @llvm.assume(i1)

; CHECK-LABEL: Alias sets for function 'arg_arg'
; CHECK: AliasSet[{{.*}}, 1] must alias, Ref Pointers: (ptr %a1, LocationSize::precise(1))
; CHECK: AliasSet[{{.*}}, 1] must alias, Ref Pointers: (ptr %a2, LocationSize::precise(1))
define void @arg_arg(ptr %a1, ptr %a2) {
entry:
  call void @llvm.assume(i1 true) [ "separate_storage"(ptr %a1, ptr %a2) ]
  %0 = load i8, ptr %a1
  %1 = load i8, ptr %a2
  ret void
}

; CHECK-LABEL: Alias sets for function 'arg_inst'
; CHECK: AliasSet[{{.*}}, 1] must alias, Ref Pointers: (ptr %a1, LocationSize::precise(1))
; CHECK: AliasSet[{{.*}}, 1] must alias, Ref Pointers: (ptr %0, LocationSize::precise(1))
define void @arg_inst(ptr %a1, ptr %a2) {
entry:
  %0 = getelementptr inbounds i8, ptr %a2, i64 20
  call void @llvm.assume(i1 true) [ "separate_storage"(ptr %a1, ptr %0) ]
  %1 = load i8, ptr %a1
  %2 = load i8, ptr %0
  ret void
}

; CHECK-LABEL: Alias sets for function 'inst_inst'
; CHECK: AliasSet[{{.*}}, 1] must alias, Ref Pointers: (ptr %0, LocationSize::precise(1))
; CHECK: AliasSet[{{.*}}, 1] must alias, Ref Pointers: (ptr %1, LocationSize::precise(1))
define void @inst_inst(ptr %a1, ptr %a2) {
entry:
  %0 = getelementptr inbounds i8, ptr %a1, i64 20
  %1 = getelementptr inbounds i8, ptr %a2, i64 20
  call void @llvm.assume(i1 true) [ "separate_storage"(ptr %0, ptr %1) ]
  %2 = load i8, ptr %0
  %3 = load i8, ptr %1
  ret void
}

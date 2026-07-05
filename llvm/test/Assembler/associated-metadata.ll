; RUN: llvm-as < %s | llvm-dis | FileCheck %s

@gv.decl = external constant [8 x i8]
@gv.def = constant [8 x i8] zeroinitializer

@gv.associated.func.decl = external addrspace(1) constant [8 x i8], !associated !0
@gv.associated.func.def = external addrspace(1) constant [8 x i8], !associated !1

@gv.associated.gv.decl = external addrspace(1) constant [8 x i8], !associated !2
@gv.associated.gv.def = external addrspace(1) constant [8 x i8], !associated !3

@alias = alias i32, ptr @gv.def

@gv.associated.alias.gv.def = external addrspace(1) constant [8 x i8], !associated !4

@gv.associated.alias.addrspacecast = external addrspace(1) constant [8 x i8], !associated !5
@alias.addrspacecast = alias i32, ptr addrspace(1) addrspacecast (ptr @gv.def to ptr addrspace(1))


@gv.def.associated.addrspacecast = external addrspace(1) constant [8 x i8], !associated !6

@ifunc = dso_local ifunc i32 (i32), ptr @ifunc_resolver
@gv.associated.ifunc = external constant [8 x i8], !associated !7

@gv.associated.null = external constant [8 x i8], !associated !8
@gv.associated.inttoptr = external constant [8 x i8], !associated !9
@gv.associated.poison = external constant [8 x i8], !associated !10
@gv.associated.undef = external constant [8 x i8], !associated !11
@associated.addrspacecast.null = external addrspace(1) constant [8 x i8], !associated !12


;.
; CHECK: @[[GV_DECL:[a-zA-Z0-9_$"\\.-]+]] = external constant [8 x i8]
; CHECK: @[[GV_DEF:[a-zA-Z0-9_$"\\.-]+]] = constant [8 x i8] zeroinitializer
; CHECK: @[[GV_ASSOCIATED_FUNC_DECL:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !0
; CHECK: @[[GV_ASSOCIATED_FUNC_DEF:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !1
; CHECK: @[[GV_ASSOCIATED_GV_DECL:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !2
; CHECK: @[[GV_ASSOCIATED_GV_DEF:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !3
; CHECK: @[[GV_ASSOCIATED_ALIAS_GV_DEF:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !4
; CHECK: @[[GV_ASSOCIATED_ALIAS_ADDRSPACECAST:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !5
; CHECK: @[[GV_DEF_ASSOCIATED_ADDRSPACECAST:[a-zA-Z0-9_$"\\.-]+]] = external addrspace(1) constant [8 x i8], !associated !6
; CHECK: @[[GV_ASSOCIATED_IFUNC:[a-zA-Z0-9_$"\\.-]+]] = external constant [8 x i8], !associated !7
; CHECK: @[[GV_ASSOCIATED_NULL:[a-zA-Z0-9_$"\\.-]+]] = external constant [8 x i8], !associated !8
; CHECK: @[[GV_ASSOCIATED_INTTOPTR:[a-zA-Z0-9_$"\\.-]+]] = external constant [8 x i8], !associated !9
; CHECK: @[[GV_ASSOCIATED_POISON:[a-zA-Z0-9_$"\\.-]+]] = external constant [8 x i8], !associated !10
; CHECK: @[[GV_ASSOCIATED_UNDEF:[a-zA-Z0-9_$"\\.-]+]] = external constant [8 x i8], !associated !11
; CHECK: @[[ALIAS:[a-zA-Z0-9_$"\\.-]+]] = alias i32, ptr @gv.def
; CHECK: @[[ALIAS_ADDRSPACECAST:[a-zA-Z0-9_$"\\.-]+]] = alias i32, addrspacecast (ptr @gv.def to ptr addrspace(1))
; CHECK: @[[IFUNC:[a-zA-Z0-9_$"\\.-]+]] = dso_local ifunc i32 (i32), ptr @ifunc_resolver
;.
define ptr @ifunc_resolver() {
; CHECK-LABEL: @ifunc_resolver(
; CHECK-NEXT:    ret ptr null
;
  ret ptr null
}


declare void @func.decl()
define void @func.def() {
; CHECK-LABEL: @func.def(
; CHECK-NEXT:    ret void
;
  ret void
}

!0 = !{ ptr @func.decl }
!1 = !{ ptr @func.def }
!2 = !{ ptr @gv.decl }
!3 = !{ ptr @gv.def }
!4 = !{ ptr @alias }
!5 = !{ ptr addrspace(1) @alias.addrspacecast }
!6 = !{ ptr addrspace(1) addrspacecast (ptr @gv.def to ptr addrspace(1)) }
!7 = !{ ptr @ifunc }
!8 = !{ ptr null }
!9 = !{ ptr inttoptr (i64 12345 to ptr) }
!10 = !{ ptr poison }
!11 = !{ ptr undef }
!12 = !{ptr addrspace(1) addrspacecast (ptr null to ptr addrspace(1))}
;.
; CHECK: [[META0:![0-9]+]] = !{ptr @func.decl}
; CHECK: [[META1:![0-9]+]] = !{ptr @func.def}
; CHECK: [[META2:![0-9]+]] = !{ptr @gv.decl}
; CHECK: [[META3:![0-9]+]] = !{ptr @gv.def}
; CHECK: [[META4:![0-9]+]] = !{ptr @alias}
; CHECK: [[META5:![0-9]+]] = !{ptr addrspace(1) @alias.addrspacecast}
; CHECK: [[META6:![0-9]+]] = !{ptr addrspace(1) addrspacecast (ptr @gv.def to ptr addrspace(1))}
; CHECK: [[META7:![0-9]+]] = !{ptr @ifunc}
; CHECK: [[META8:![0-9]+]] = !{ptr null}
; CHECK: [[META9:![0-9]+]] = !{ptr inttoptr (i64 12345 to ptr)}
; CHECK: [[META10:![0-9]+]] = !{ptr poison}
; CHECK: [[META11:![0-9]+]] = !{ptr undef}
;.

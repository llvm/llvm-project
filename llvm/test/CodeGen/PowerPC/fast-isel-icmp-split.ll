; RUN: llc -verify-machineinstrs -O0 -relocation-model=pic < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux"

%"class.std::__1::__tree_node.130.151" = type { %"class.std::__1::__tree_node_base.base.128.149", ptr }
%"class.std::__1::__tree_node_base.base.128.149" = type <{ %"class.std::__1::__tree_end_node.127.148", ptr, ptr, i8 }>
%"class.std::__1::__tree_end_node.127.148" = type { ptr }
%"class.std::__1::__tree_node_base.126.147" = type <{ %"class.std::__1::__tree_end_node.127.148", ptr, ptr, i8, [7 x i8] }>
%"class.boost::serialization::extended_type_info.129.150" = type { ptr, i32, ptr }

; Function Attrs: noinline
define void @_ZN5boost13serialization18extended_type_info4findEPKc() #0 align 2 personality ptr @__gxx_personality_v0 {
entry:
  br i1 undef, label %cond.true, label %cond.false

; CHECK: @_ZN5boost13serialization18extended_type_info4findEPKc

cond.true:                                        ; preds = %entry
  br label %cond.end

cond.false:                                       ; preds = %entry
  unreachable
                                                  ; No predecessors!
  br label %cond.end

cond.end:                                         ; preds = %0, %cond.true
  invoke void @_ZNKSt3__16__treeIPKN5boost13serialization18extended_type_infoENS2_6detail11key_compareENS_9allocatorIS5_EEE4findIS5_EENS_21__tree_const_iteratorIS5_PNS_11__tree_nodeIS5_PvEElEERKT_()
          to label %_ZNKSt3__18multisetIPKN5boost13serialization18extended_type_infoENS2_6detail11key_compareENS_9allocatorIS5_EEE4findERKS5_.exit unwind label %lpad

_ZNKSt3__18multisetIPKN5boost13serialization18extended_type_infoENS2_6detail11key_compareENS_9allocatorIS5_EEE4findERKS5_.exit: ; preds = %cond.end
  br label %invoke.cont

invoke.cont:                                      ; preds = %_ZNKSt3__18multisetIPKN5boost13serialization18extended_type_infoENS2_6detail11key_compareENS_9allocatorIS5_EEE4findERKS5_.exit
  %1 = load ptr, ptr undef, align 8
  %cmp.i = icmp eq ptr undef, %1
  br label %invoke.cont.2

invoke.cont.2:                                    ; preds = %invoke.cont
  br i1 %cmp.i, label %if.then, label %if.end

if.then:                                          ; preds = %invoke.cont.2
  br label %cleanup

lpad:                                             ; preds = %cond.end
  %2 = landingpad { ptr, i32 }
          cleanup
  br label %eh.resume

if.end:                                           ; preds = %invoke.cont.2
  br label %invoke.cont.4

invoke.cont.4:                                    ; preds = %if.end
  br label %cleanup

cleanup:                                          ; preds = %invoke.cont.4, %if.then
  ret void

eh.resume:                                        ; preds = %lpad
  resume { ptr, i32 } undef
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noinline
declare void @_ZNKSt3__16__treeIPKN5boost13serialization18extended_type_infoENS2_6detail11key_compareENS_9allocatorIS5_EEE4findIS5_EENS_21__tree_const_iteratorIS5_PNS_11__tree_nodeIS5_PvEElEERKT_() #0 align 2

attributes #0 = { noinline "target-cpu"="a2q" }

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"PIC Level", i32 2}


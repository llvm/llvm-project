; RUN: llc -verify-machineinstrs -disable-ppc-unaligned < %s | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%class.spell_checker.21.103.513.538 = type { %"class.std::map.20.102.512.537" }
%"class.std::map.20.102.512.537" = type { %"class.std::_Rb_tree.19.101.511.536" }
%"class.std::_Rb_tree.19.101.511.536" = type { %"struct.std::_Rb_tree<std::pair<const char *, const char *>, std::pair<const std::pair<const char *, const char *>, int>, std::_Select1st<std::pair<const std::pair<const char *, const char *>, int>>, std::less<std::pair<const char *, const char *>>, std::allocator<std::pair<const std::pair<const char *, const char *>, int>> >::_Rb_tree_impl.18.100.510.535" }
%"struct.std::_Rb_tree<std::pair<const char *, const char *>, std::pair<const std::pair<const char *, const char *>, int>, std::_Select1st<std::pair<const std::pair<const char *, const char *>, int>>, std::less<std::pair<const char *, const char *>>, std::allocator<std::pair<const std::pair<const char *, const char *>, int>> >::_Rb_tree_impl.18.100.510.535" = type { %"struct.std::less.16.98.508.533", %"struct.std::_Rb_tree_node_base.17.99.509.534", i64 }
%"struct.std::less.16.98.508.533" = type { i8 }
%"struct.std::_Rb_tree_node_base.17.99.509.534" = type { i32, ptr, ptr, ptr }

define void @test1(ptr %this) unnamed_addr align 2 {
entry:
  %_M_header.i.i.i.i.i.i = getelementptr inbounds %class.spell_checker.21.103.513.538, ptr %this, i64 0, i32 0, i32 0, i32 0, i32 1
  call void @llvm.memset.p0.i64(ptr align 4 %_M_header.i.i.i.i.i.i, i8 0, i64 40, i1 false) nounwind
  store ptr %_M_header.i.i.i.i.i.i, ptr undef, align 8
  unreachable
}

; CHECK: @test1
; CHECK: stwu

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) nounwind

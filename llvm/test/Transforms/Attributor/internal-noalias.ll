; RUN: opt -S -passes=attributor -aa-pipeline='basic-aa' -attributor-disable=false -attributor-max-iterations-verify -attributor-annotate-decl-cs -attributor-max-iterations=6 < %s | FileCheck %s

define dso_local i32 @visible(i32* noalias %A, i32* noalias %B) #0 {
entry:
  %call1 = call i32 @noalias_args(i32* %A, i32* %B)
  %call2 = call i32 @noalias_args_argmem(i32* %A, i32* %B)
  %add = add nsw i32 %call1, %call2
  ret i32 %add
}

; CHECK: define private i32 @noalias_args(i32* nocapture nofree nonnull readonly align 4 dereferenceable(4) %A, i32* noalias nocapture nofree nonnull readonly align 4 dereferenceable(4) %B)

define private i32 @noalias_args(i32* %A, i32* %B) #0 {
entry:
  %0 = load i32, i32* %A, align 4
  %1 = load i32, i32* %B, align 4
  %add = add nsw i32 %0, %1
  %call = call i32 @noalias_args_argmem(i32* %A, i32* %B)
  %add2 = add nsw i32 %add, %call
  ret i32 %add2
}


; CHECK: define internal i32 @noalias_args_argmem(i32* nocapture nofree nonnull readonly align 4 dereferenceable(4) %A, i32* noalias nocapture nofree nonnull readonly align 4 dereferenceable(4) %B)
define internal i32 @noalias_args_argmem(i32* %A, i32* %B) #1 {
entry:
  %0 = load i32, i32* %A, align 4
  %1 = load i32, i32* %B, align 4
  %add = add nsw i32 %0, %1
  ret i32 %add
}

define dso_local i32 @visible_local(i32* %A) #0 {
entry:
  %B = alloca i32, align 4
  store i32 5, i32* %B, align 4
  %call1 = call i32 @noalias_args(i32* %A, i32* nonnull %B)
  %call2 = call i32 @noalias_args_argmem(i32* %A, i32* nonnull %B)
  %add = add nsw i32 %call1, %call2
  ret i32 %add
}

; CHECK: define internal i32 @noalias_args_argmem_ro(i32 %0, i32 %1)
define internal i32 @noalias_args_argmem_ro(i32* %A, i32* %B) #1 {
  %t0 = load i32, i32* %A, align 4
  %t1 = load i32, i32* %B, align 4
  %add = add nsw i32 %t0, %t1
  ret i32 %add
}

define i32 @visible_local_2() {
  %B = alloca i32, align 4
  store i32 5, i32* %B, align 4
  %call = call i32 @noalias_args_argmem_ro(i32* %B, i32* %B)
  ret i32 %call
}

; CHECK: define internal i32 @noalias_args_argmem_rn(i32* noalias nocapture nofree nonnull align 4 dereferenceable(4) %B)
define internal i32 @noalias_args_argmem_rn(i32* %A, i32* %B) #1 {
  %t0 = load i32, i32* %B, align 4
  store i32 0, i32* %B
  ret i32 %t0
}

define i32 @visible_local_3() {
  %B = alloca i32, align 4
  store i32 5, i32* %B, align 4
  %call = call i32 @noalias_args_argmem_rn(i32* %B, i32* %B)
  ret i32 %call
}

attributes #0 = { noinline nounwind uwtable willreturn }
attributes #1 = { argmemonly noinline nounwind uwtable willreturn}

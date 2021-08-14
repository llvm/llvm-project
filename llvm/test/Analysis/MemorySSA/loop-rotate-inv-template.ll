; RUN: opt -disable-output -loop-rotate -verify-memoryssa %s
; REQUIRES: asserts

; Function Attrs: nounwind
define dso_local void @bar() local_unnamed_addr #0 align 32 {
entry:
  br label %looplabel.exit.i

looplabel.exit.i: ; preds = %if.end.i, %entry
  %0 = phi i1 (i32*, i32*)* [ @foo, %entry ], [ undef, %if.end.i ]
  %call3.i.i = call zeroext i1 %0(i32* nonnull dereferenceable(16) undef, i32* nonnull undef)
  br i1 %call3.i.i, label %if.end.i, label %label.exit

if.end.i:                                         ; preds = %looplabel.exit.i
  %tobool.i = icmp eq i32* undef, null
  br label %looplabel.exit.i

label.exit: ; preds = %looplabel.exit.i
  ret void
}

; Function Attrs: readonly
declare dso_local i1 @foo(i32*, i32*) #1 align 32

attributes #0 = { nounwind }
attributes #1 = { readonly }


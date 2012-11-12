; RUN: opt < %s -instcombine -disable-output

%opaque_struct = type opaque

@G = external global [0 x %opaque_struct]

declare void @foo(%opaque_struct*)

define void @bar() {
  call void @foo(%opaque_struct* bitcast ([0 x %opaque_struct]* @G to %opaque_struct*))
  ret void
}

%struct.A = type { %struct.B }
%struct.B = type opaque

define i32 @foo(%struct.A** %A) {
  ret i32 0
}

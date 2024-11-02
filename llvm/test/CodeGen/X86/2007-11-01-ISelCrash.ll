; RUN: llc < %s -mtriple=i686--

        %"struct.K::JL" = type <{ i8 }>
        %struct.jv = type { i64 }

declare fastcc i64 @f(i32, ptr, ptr, ptr, ptr)

define void @t(ptr %obj, ptr %name, ptr %sig, ptr %args) {
entry:
        %tmp5 = tail call fastcc i64 @f( i32 1, ptr %obj, ptr %name, ptr %sig, ptr %args )         ; <i64> [#uses=0]
        ret void
}

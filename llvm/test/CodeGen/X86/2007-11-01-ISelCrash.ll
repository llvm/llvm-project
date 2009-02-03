; RUN: llvm-as < %s | llc -march=x86

        %"struct.K::JL" = type <{ i8 }>
        %struct.jv = type { i64 }

declare fastcc i64 @f(i32, %"struct.K::JL"*, i8*, i8*, %struct.jv*)

define void @t(%"struct.K::JL"* %obj, i8* %name, i8* %sig, %struct.jv* %args) {
entry:
        %tmp5 = tail call fastcc i64 @f( i32 1, %"struct.K::JL"* %obj, i8* %name, i8* %sig, %struct.jv* %args )         ; <i64> [#uses=0]
        ret void
}

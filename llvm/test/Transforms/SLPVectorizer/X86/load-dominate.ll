; RUN: opt -slp-vectorizer -S -mtriple=x86_64-unknown-linux-gnu < %s

%class.1 = type { %class.2 }
%class.2 = type { %"class.3" }
%"class.3" = type { %"struct.1", i64 }
%"struct.1" = type { [8 x i64] }

$_ZN1C10SwitchModeEv = comdat any

; Function Attrs: uwtable
define void @_ZN1C10SwitchModeEv() local_unnamed_addr #0 comdat align 2 {
for.body.lr.ph.i:
  %or.1 = or i64 undef, 1
  store i64 %or.1, i64* undef, align 8
  %foo.1 = getelementptr inbounds %class.1, %class.1* undef, i64 0, i32 0, i32 0, i32 0, i32 0, i64 0
  %foo.3 = load i64, i64* %foo.1, align 8
  %foo.2 = getelementptr inbounds %class.1, %class.1* undef, i64 0, i32 0, i32 0, i32 0, i32 0, i64 1
  %foo.4 = load i64, i64* %foo.2, align 8
  %bar5 = load i64, i64* undef, align 8
  %and.2 = and i64 %or.1, %foo.3
  %and.1 = and i64 %bar5, %foo.4
  %bar3 = getelementptr inbounds %class.2, %class.2* undef, i64 0, i32 0, i32 0, i32 0, i64 0
  store i64 %and.2, i64* %bar3, align 8
  %bar4 = getelementptr inbounds %class.2, %class.2* undef, i64 0, i32 0, i32 0, i32 0, i64 1
  store i64 %and.1, i64* %bar4, align 8
  ret void
}

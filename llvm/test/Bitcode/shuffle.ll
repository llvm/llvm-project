; RUN: llvm-as < %s | llvm-dis

; <rdar://problem/8622574>
; tests the bitcodereader can handle the case where the reader will initially
; create shuffle with a place holder mask.


define <4 x float> @test(<2 x double> %d2)  {
entry:
  %call20.i = tail call <4 x float> @cmp(<2 x double> %d2,
                                        <2 x double> bitcast (
                                          <4 x float> shufflevector (
                                            <3 x float> shufflevector (
                                              <4 x float> shufflevector (
                                                <3 x float> bitcast (
                                                  i96 trunc (
                                                    i128 bitcast (<2 x double> bitcast (
                                                      <4 x i32> <i32 0, i32 0, i32 0, i32 undef> to <2 x double>)
                                                    to i128) to i96)
                                                  to <3 x float>),
                                                <3 x float> undef,
                                                <4 x i32> <i32 0, i32 1, i32 2, i32 undef>),
                                              <4 x float> undef,
                                            <3 x i32> <i32 0, i32 1, i32 2>),
                                            <3 x float> undef,
                                            <4 x i32> <i32 0, i32 1, i32 2, i32 undef>)
                                          to <2 x double>))
  ret <4 x float> %call20.i
}

declare <4 x float> @cmp(<2 x double>, <2 x double>)

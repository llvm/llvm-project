; Test vector shuffles on vectors with implicitly extended elements
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-CODE %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | \
; RUN:   FileCheck -check-prefix=CHECK-VECTOR %s

define <4 x i16> @f1(<4 x i16> %x) {
; CHECK-CODE-LABEL: f1:
; CHECK-CODE: larl [[REG:%r[0-5]]],
; CHECK-CODE: vl [[MASK:%v[0-9]+]], 0([[REG]])
; CHECK-CODE: vgbm [[ELT:%v[0-9]+]], 0
; CHECK-CODE: vperm %v24, %v24, [[ELT]], [[MASK]]
; CHECK-CODE: br %r14

; CHECK-VECTOR: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .byte   6
; CHECK-VECTOR-NEXT: .byte   7
; CHECK-VECTOR-NEXT: .byte   22
; CHECK-VECTOR-NEXT: .byte   23
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        
; CHECK-VECTOR-NEXT: .space  1                                        

  %elt = extractelement <4 x i16> %x, i32 3
  %vec1 = insertelement <4 x i16> undef, i16 %elt, i32 2
  %vec2 = insertelement <4 x i16> %vec1, i16 0, i32 3
  ret <4 x i16> %vec2
}


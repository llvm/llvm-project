// RUN: mlir-opt %s -convert-pdl-to-pdl-interp --verify-diagnostics

func.func private @pattern_body() -> (!pdl.type, !pdl.type, !pdl.operation)
func.func private @rewrite_body(!pdl.type, !pdl.type, !pdl.operation)

// expected-error@below {{pdl_interp backend does not support non-materializable patterns}}
pdl.pattern @nonmaterializable_pattern : benefit(1) nonmaterializable {
  %type1, %type2, %root = func.call @pattern_body()
    : () -> (!pdl.type, !pdl.type, !pdl.operation)
  rewrite %root {
    func.call @rewrite_body(%type1, %type2, %root)
      : (!pdl.type, !pdl.type, !pdl.operation) -> ()
  }
}

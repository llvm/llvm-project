// RUN: mlir-opt %s -test-external-side-effects -verify-diagnostics

func.func @external_side_effects() {
  // effects attr present but empty: hasKnownMemoryEffects = true, no effects.
  // expected-remark@+2 {{implements MemoryEffectOpInterface: true}}
  // expected-remark@+1 {{operation has no memory effects}}
  %0 = "test.external_side_effect_op"() {effects = []} : () -> i32

  // No effects attr: hasKnownMemoryEffects = false, isa<> returns false.
  // expected-remark@+1 {{implements MemoryEffectOpInterface: false}}
  %1 = "test.external_side_effect_op"() {} : () -> i32

  // effects attr with read+write: hasKnownMemoryEffects = true, known effects.
  // expected-remark@+3 {{implements MemoryEffectOpInterface: true}}
  // expected-remark@+2 {{found an instance of 'read' on resource '<Default>'}}
  // expected-remark@+1 {{found an instance of 'write' on resource '<Default>'}}
  %2 = "test.external_side_effect_op"() {effects = [
    {effect="read"}, {effect="write"}
  ]} : () -> i32

  func.return
}

# Vector unit-strde load/store.

vsetvli zero, zero, e8, mf8, ta, ma
vle8.v   v8, (a0)
vsetvli zero, zero, e8, mf4, ta, ma
vle8.v   v8, (a0)
vsetvli zero, zero, e8, mf2, ta, ma
vle8.v   v8, (a0)
vsetvli zero, zero, e8, m1, ta, ma
vle8.v   v8, (a0)
vsetvli zero, zero, e8, m2, ta, ma
vle8.v   v8, (a0)
vsetvli zero, zero, e8, m4, ta, ma
vle8.v   v8, (a0)
vsetvli zero, zero, e8, m8, ta, ma
vle8.v   v8, (a0)

vsetvli zero, zero, e16, mf4, ta, ma
vle16.v   v8, (a0)
vsetvli zero, zero, e16, mf2, ta, ma
vle16.v   v8, (a0)
vsetvli zero, zero, e16, m1, ta, ma
vle16.v   v8, (a0)
vsetvli zero, zero, e16, m2, ta, ma
vle16.v   v8, (a0)
vsetvli zero, zero, e16, m4, ta, ma
vle16.v   v8, (a0)
vsetvli zero, zero, e16, m8, ta, ma
vle16.v   v8, (a0)

vsetvli zero, zero, e32, mf2, ta, ma
vle32.v   v8, (a0)
vsetvli zero, zero, e32, m1, ta, ma
vle32.v   v8, (a0)
vsetvli zero, zero, e32, m2, ta, ma
vle32.v   v8, (a0)
vsetvli zero, zero, e32, m4, ta, ma
vle32.v   v8, (a0)
vsetvli zero, zero, e32, m8, ta, ma
vle32.v   v8, (a0)

vsetvli zero, zero, e64, m1, ta, ma
vle64.v   v8, (a0)
vsetvli zero, zero, e64, m2, ta, ma
vle64.v   v8, (a0)
vsetvli zero, zero, e64, m4, ta, ma
vle64.v   v8, (a0)
vsetvli zero, zero, e64, m8, ta, ma
vle64.v   v8, (a0)

vsetvli zero, zero, e8, mf8, ta, ma
vse8.v   v8, (a0)
vsetvli zero, zero, e8, mf4, ta, ma
vse8.v   v8, (a0)
vsetvli zero, zero, e8, mf2, ta, ma
vse8.v   v8, (a0)
vsetvli zero, zero, e8, m1, ta, ma
vse8.v   v8, (a0)
vsetvli zero, zero, e8, m2, ta, ma
vse8.v   v8, (a0)
vsetvli zero, zero, e8, m4, ta, ma
vse8.v   v8, (a0)
vsetvli zero, zero, e8, m8, ta, ma
vse8.v   v8, (a0)

vsetvli zero, zero, e16, mf4, ta, ma
vse16.v   v8, (a0)
vsetvli zero, zero, e16, mf2, ta, ma
vse16.v   v8, (a0)
vsetvli zero, zero, e16, m1, ta, ma
vse16.v   v8, (a0)
vsetvli zero, zero, e16, m2, ta, ma
vse16.v   v8, (a0)
vsetvli zero, zero, e16, m4, ta, ma
vse16.v   v8, (a0)
vsetvli zero, zero, e16, m8, ta, ma
vse16.v   v8, (a0)

vsetvli zero, zero, e32, mf2, ta, ma
vse32.v   v8, (a0)
vsetvli zero, zero, e32, m1, ta, ma
vse32.v   v8, (a0)
vsetvli zero, zero, e32, m2, ta, ma
vse32.v   v8, (a0)
vsetvli zero, zero, e32, m4, ta, ma
vse32.v   v8, (a0)
vsetvli zero, zero, e32, m8, ta, ma
vse32.v   v8, (a0)

vsetvli zero, zero, e64, m1, ta, ma
vse64.v   v8, (a0)
vsetvli zero, zero, e64, m2, ta, ma
vse64.v   v8, (a0)
vsetvli zero, zero, e64, m4, ta, ma
vse64.v   v8, (a0)
vsetvli zero, zero, e64, m8, ta, ma
vse64.v   v8, (a0)

# Unit-stride mask load/store

vsetvli zero, zero, e8, mf8, ta, ma
vlm.v     v8, (a0)
vsetvli zero, zero, e8, mf4, ta, ma
vlm.v     v8, (a0)
vsetvli zero, zero, e8, mf2, ta, ma
vlm.v     v8, (a0)
vsetvli zero, zero, e8, m1, ta, ma
vlm.v     v8, (a0)
vsetvli zero, zero, e8, m2, ta, ma
vlm.v     v8, (a0)
vsetvli zero, zero, e8, m4, ta, ma
vlm.v     v8, (a0)
vsetvli zero, zero, e8, m8, ta, ma
vlm.v     v8, (a0)

vsetvli zero, zero, e8, mf8, ta, ma
vsm.v     v8, (a0)
vsetvli zero, zero, e8, mf4, ta, ma
vsm.v     v8, (a0)
vsetvli zero, zero, e8, mf2, ta, ma
vsm.v     v8, (a0)
vsetvli zero, zero, e8, m1, ta, ma
vsm.v     v8, (a0)
vsetvli zero, zero, e8, m2, ta, ma
vsm.v     v8, (a0)
vsetvli zero, zero, e8, m4, ta, ma
vsm.v     v8, (a0)
vsetvli zero, zero, e8, m8, ta, ma
vsm.v     v8, (a0)

# Fault-only-first

vsetvli zero, zero, e8, mf8, ta, ma
vle8ff.v   v8, (a0)
vsetvli zero, zero, e8, mf4, ta, ma
vle8ff.v   v8, (a0)
vsetvli zero, zero, e8, mf2, ta, ma
vle8ff.v   v8, (a0)
vsetvli zero, zero, e8, m1, ta, ma
vle8ff.v   v8, (a0)
vsetvli zero, zero, e8, m2, ta, ma
vle8ff.v   v8, (a0)
vsetvli zero, zero, e8, m4, ta, ma
vle8ff.v   v8, (a0)
vsetvli zero, zero, e8, m8, ta, ma
vle8ff.v   v8, (a0)

vsetvli zero, zero, e16, mf4, ta, ma
vle16ff.v   v8, (a0)
vsetvli zero, zero, e16, mf2, ta, ma
vle16ff.v   v8, (a0)
vsetvli zero, zero, e16, m1, ta, ma
vle16ff.v   v8, (a0)
vsetvli zero, zero, e16, m2, ta, ma
vle16ff.v   v8, (a0)
vsetvli zero, zero, e16, m4, ta, ma
vle16ff.v   v8, (a0)
vsetvli zero, zero, e16, m8, ta, ma
vle16ff.v   v8, (a0)

vsetvli zero, zero, e32, mf2, ta, ma
vle32ff.v   v8, (a0)
vsetvli zero, zero, e32, m1, ta, ma
vle32ff.v   v8, (a0)
vsetvli zero, zero, e32, m2, ta, ma
vle32ff.v   v8, (a0)
vsetvli zero, zero, e32, m4, ta, ma
vle32ff.v   v8, (a0)
vsetvli zero, zero, e32, m8, ta, ma
vle32ff.v   v8, (a0)

vsetvli zero, zero, e64, m1, ta, ma
vle64ff.v   v8, (a0)
vsetvli zero, zero, e64, m2, ta, ma
vle64ff.v   v8, (a0)
vsetvli zero, zero, e64, m4, ta, ma
vle64ff.v   v8, (a0)
vsetvli zero, zero, e64, m8, ta, ma
vle64ff.v   v8, (a0)

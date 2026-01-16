# Input instructions for the 'Zvbc' extension.
# Note: These instructions only work with EEW=64

vsetvli zero, zero, e64, m1, tu, mu
vclmul.vv v4, v8, v12
vclmul.vx v4, v8, a0
vclmulh.vv v4, v8, v12
vclmulh.vx v4, v8, a0

vsetvli zero, zero, e64, m2, tu, mu
vclmul.vv v4, v8, v12
vclmul.vx v4, v8, a0
vclmulh.vv v4, v8, v12
vclmulh.vx v4, v8, a0

vsetvli zero, zero, e64, m4, tu, mu
vclmul.vv v4, v8, v12
vclmul.vx v4, v8, a0
vclmulh.vv v4, v8, v12
vclmulh.vx v4, v8, a0

vsetvli zero, zero, e64, m8, tu, mu
vclmul.vv  v8, v12, v24
vclmul.vx  v8, v12, a0
vclmulh.vv v8, v12, v24
vclmulh.vx v8, v12, a0

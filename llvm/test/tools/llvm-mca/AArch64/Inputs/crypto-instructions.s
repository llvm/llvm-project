aesd z0.b, z0.b, z31.b
aese z0.b, z0.b, z31.b
aesimc z0.b, z0.b
aesmc z0.b, z0.b

sha1h s0, s1
sha1su1 v0.4s, v1.4s
sha256su0 v0.4s, v1.4s
sha1c q0, s1, v2.4s
sha1p q0, s1, v2.4s
sha1m q0, s1, v2.4s
sha1su0 v0.4s, v1.4s, v2.4s
sha256h q0, q1, v2.4s
sha256h2 q0, q1, v2.4s
sha256su1 v0.4s, v1.4s, v2.4s

// armv8.2a
sha512h   q0, q1, v2.2d
sha512h2  q0, q1, v2.2d
sha512su0 v11.2d, v12.2d
sha512su1 v11.2d, v13.2d, v14.2d
eor3  v25.16b, v12.16b, v7.16b, v2.16b
rax1  v30.2d, v29.2d, v26.2d
xar v26.2d, v21.2d, v27.2d, #63
bcax  v31.16b, v26.16b, v2.16b, v1.16b
sm3ss1  v20.4s, v23.4s, v21.4s, v22.4s
sm3tt1a v20.4s, v23.4s, v21.s[3]
sm3tt1b v20.4s, v23.4s, v21.s[3]
sm3tt2a v20.4s, v23.4s, v21.s[3]
sm3tt2b v20.4s, v23.4s, v21.s[3]
sm3partw1 v30.4s, v29.4s, v26.4s
sm3partw2 v30.4s, v29.4s, v26.4s
sm4ekey v11.4s, v11.4s, v19.4s
sm4e  v2.4s, v15.4s

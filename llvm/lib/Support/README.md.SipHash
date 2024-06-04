# SipHash

[![License:
CC0-1.0](https://licensebuttons.net/l/zero/1.0/80x15.png)](http://creativecommons.org/publicdomain/zero/1.0/)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


SipHash is a family of pseudorandom functions (PRFs) optimized for speed on short messages.
This is the reference C code of SipHash: portable, simple, optimized for clarity and debugging.

SipHash was designed in 2012 by [Jean-Philippe Aumasson](https://aumasson.jp)
and [Daniel J. Bernstein](https://cr.yp.to) as a defense against [hash-flooding
DoS attacks](https://aumasson.jp/siphash/siphashdos_29c3_slides.pdf).

SipHash is:

* *Simpler and faster* on short messages than previous cryptographic
algorithms, such as MACs based on universal hashing.

* *Competitive in performance* with insecure non-cryptographic algorithms, such as [fhhash](https://github.com/cbreeden/fxhash).

* *Cryptographically secure*, with no sign of weakness despite multiple [cryptanalysis](https://eprint.iacr.org/2019/865) [projects](https://eprint.iacr.org/2019/865) by leading cryptographers.

* *Battle-tested*, with successful integration in OSs (Linux kernel, OpenBSD,
FreeBSD, FreeRTOS), languages (Perl, Python, Ruby, etc.), libraries (OpenSSL libcrypto,
Sodium, etc.) and applications (Wireguard, Redis, etc.).

As a secure pseudorandom function (a.k.a. keyed hash function), SipHash can also be used as a secure message authentication code (MAC).
But SipHash is *not a hash* in the sense of general-purpose key-less hash function such as BLAKE3 or SHA-3.
SipHash should therefore always be used with a secret key in order to be secure.


## Variants

The default SipHash is *SipHash-2-4*: it takes a 128-bit key, does 2 compression
rounds, 4 finalization rounds, and returns a 64-bit tag.

Variants can use a different number of rounds. For example, we proposed *SipHash-4-8* as a conservative version.

The following versions are not described in the paper but were designed and analyzed to fulfill applications' needs:

* *SipHash-128* returns a 128-bit tag instead of 64-bit. Versions with specified number of rounds are SipHash-2-4-128, SipHash4-8-128, and so on.

* *HalfSipHash* works with 32-bit words instead of 64-bit, takes a 64-bit key,
and returns 32-bit or 64-bit tags. For example, HalfSipHash-2-4-32 has 2
compression rounds, 4 finalization rounds, and returns a 32-bit tag.


## Security

(Half)SipHash-*c*-*d* with *c* ≥ 2 and *d* ≥ 4 is expected to provide the maximum PRF
security for any function with the same key and output size.

The standard PRF security goal allow the attacker access to the output of SipHash on messages chosen adaptively by the attacker.

Security is limited by the key size (128 bits for SipHash), such that
attackers searching 2<sup>*s*</sup> keys have chance 2<sup>*s*−128</sup> of finding
the SipHash key. 
Security is also limited by the output size. In particular, when
SipHash is used as a MAC, an attacker who blindly tries 2<sup>*s*</sup> tags will
succeed with probability 2<sup>*s*-*t*</sup>, if *t* is that tag's bit size.


## Research

* [Research paper](https://www.aumasson.jp/siphash/siphash.pdf) "SipHash: a fast short-input PRF" (accepted at INDOCRYPT 2012)
* [Slides](https://cr.yp.to/talks/2012.12.12/slides.pdf) of the presentation of SipHash at INDOCRYPT 2012 (Bernstein)
* [Slides](https://www.aumasson.jp/siphash/siphash_slides.pdf) of the presentation of SipHash at the DIAC workshop (Aumasson)


## Usage

Running

```sh
  make
```

will build tests for 

* SipHash-2-4-64
* SipHash-2-4-128
* HalfSipHash-2-4-32
* HalfSipHash-2-4-64


```C
  ./test
```

verifies 64 test vectors, and

```C
  ./debug
```

does the same and prints intermediate values.

The code can be adapted to implement SipHash-*c*-*d*, the version of SipHash
with *c* compression rounds and *d* finalization rounds, by defining `cROUNDS`
or `dROUNDS` when compiling.  This can be done with `-D` command line arguments
to many compilers such as below.

```sh
gcc -Wall --std=c99 -DcROUNDS=2 -DdROUNDS=4 siphash.c halfsiphash.c test.c -o test
```

The `makefile` also takes *c* and *d* rounds values as parameters.

```sh
make cROUNDS=2 dROUNDS=4
``` 

Obviously, if the number of rounds is modified then the test vectors
won't verify.

## Intellectual property

This code is copyright (c) 2014-2023 Jean-Philippe Aumasson, Daniel J.
Bernstein. It is multi-licensed under

* [CC0](./LICENCE_CC0)
* [MIT](./LICENSE_MIT).
* [Apache 2.0 with LLVM exceptions](./LICENSE_A2LLVM).


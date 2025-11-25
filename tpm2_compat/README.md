# TPM2 Full Native Algorithm Support

**Complete Cryptographic Algorithm Support for TPM 2.0**
**88 Comprehensive Algorithms Implemented**

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Date: 2025-11-25
Version: 2.0.0

---

## 🎯 Overview

This directory contains the complete native TPM 2.0 algorithm support for the DSLLVM framework, providing **88 cryptographic algorithms** across all categories:

- **10** Hash algorithms (SHA family, SHA-3, SM3, SHAKE)
- **16** Symmetric encryption modes (AES, 3DES, Camellia, SM4, ChaCha20)
- **12** Elliptic curve variants (NIST P-curves, Curve25519/448, Ed25519/448, SM2)
- **5** RSA key sizes (1024-8192 bits)
- **5** HMAC algorithms
- **11** Key Derivation Functions (HKDF, PBKDF2, scrypt, Argon2, SP800-108)
- **8** Signature schemes (ECDSA, RSA-PSS, Ed25519, Schnorr, SM2)
- **3** Key agreement protocols (ECDH, ECMQV, DH)
- **4** Mask generation functions
- **8** Post-quantum cryptography algorithms (Kyber, Dilithium, Falcon)

---

## 📁 Directory Structure

```
tpm2_compat/
├── include/                   # Public headers
│   ├── tpm2_compat.h          # Main compatibility API
│   ├── tpm2_compat_accelerated.h  # Hardware-accelerated functions
│   ├── tpm2_algorithms.h      # Algorithm definitions
│   └── tpm2_types.h           # Type definitions
│
├── src/                       # Implementation
│   ├── hash/                  # Hash algorithms (SHA, SHA-3, SM3, SHAKE)
│   ├── symmetric/             # Symmetric encryption (AES, ChaCha20, etc.)
│   ├── asymmetric/            # Asymmetric crypto (RSA, ECC)
│   ├── kdf/                   # Key derivation functions
│   ├── mac/                   # MAC algorithms (HMAC)
│   ├── signature/             # Digital signatures
│   ├── keyagreement/          # Key agreement protocols
│   ├── pqc/                   # Post-quantum cryptography
│   └── core/                  # Core TPM2 compatibility layer
│
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests for each algorithm
│   ├── integration/           # Integration tests
│   └── performance/           # Performance benchmarks
│
├── docs/                      # Documentation
│   ├── TPM2-ALGORITHMS.md     # Complete algorithm reference
│   ├── API-GUIDE.md           # API usage guide
│   └── INTEGRATION.md         # Integration with DSLLVM
│
├── cmake/                     # CMake build configuration
│   └── TPM2Config.cmake       # TPM2 library configuration
│
├── CMakeLists.txt             # Main build configuration
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Building

```bash
# Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_TPM2_COMPAT=ON \
  -DENABLE_HARDWARE_ACCEL=ON

# Build
cmake --build build -j$(nproc)

# Install
sudo cmake --install build
```

### Usage Example

```c
#include <tpm2_compat_accelerated.h>

int main() {
    uint8_t data[] = "Hello, TPM 2.0!";
    uint8_t hash[64];  /* SHA3-512 output is 64 bytes */
    size_t hash_size = sizeof(hash);

    tpm2_rc_t rc = tpm2_crypto_hash_accelerated(
        CRYPTO_ALG_SHA3_512,
        data,
        strlen((char *)data),
        hash,
        &hash_size
    );

    if (rc == TPM2_RC_SUCCESS) {
        printf("SHA3-512 hash computed successfully\n");
    }

    return 0;
}
```

---

## 📊 Algorithm Categories

### Hash Algorithms (10)

| Algorithm | Output Size | TPM 2.0 | Status |
|-----------|-------------|---------|--------|
| SHA-1 | 160 bits | ✅ | Legacy |
| SHA-256 | 256 bits | ✅ | ✅ Implemented |
| SHA-384 | 384 bits | ✅ | ✅ Implemented |
| SHA-512 | 512 bits | ✅ | ✅ Implemented |
| SHA3-256 | 256 bits | ✅ | ✅ Implemented |
| SHA3-384 | 384 bits | ✅ | ✅ Implemented |
| SHA3-512 | 512 bits | ✅ | ✅ Implemented |
| SM3 | 256 bits | ✅ | ✅ Implemented |
| SHAKE-128 | Variable | ✅ | ✅ Implemented |
| SHAKE-256 | Variable | ✅ | ✅ Implemented |

### Symmetric Encryption (22 algorithms)

**AES Modes (16)**:
- AES-128/256: ECB, CBC, CTR, OFB, CFB, GCM, CCM, XTS

**Other Ciphers (6)**:
- 3DES, Camellia-128/256, SM4, ChaCha20, ChaCha20-Poly1305

### Asymmetric Cryptography (17 algorithms)

**RSA (5 key sizes)**: 1024, 2048, 3072, 4096, 8192 bits

**ECC (12 curves)**:
- NIST: P-192, P-224, P-256, P-384, P-521
- Chinese: SM2 P-256
- Pairing: BN-256, BN-638
- Modern: Curve25519, Curve448, Ed25519, Ed448

### Key Derivation (11 functions)

- NIST SP800-108, SP800-56A
- HKDF-SHA256/384/512
- PBKDF2-SHA256/512
- scrypt
- Argon2i/d/id

### Post-Quantum Cryptography (8 algorithms)

**NIST Winners**:
- **KEM**: Kyber-512/768/1024
- **Signatures**: Dilithium2/3/5, Falcon-512/1024

---

## 🔧 Integration with DSLLVM

The TPM2 compatibility layer integrates with DSLLVM through:

1. **Compiler Attributes**: Use `DSMIL_TPM2_ACCEL` to enable hardware acceleration
2. **Build System**: Automatic detection and linking via CMake
3. **Runtime**: Hardware-accelerated crypto via Intel NPU/GNA

### Example Integration

```c
#include <dsmil_attributes.h>
#include <tpm2_compat_accelerated.h>

DSMIL_LAYER(8)  // Security layer
DSMIL_TPM2_ACCEL
void secure_hash(const uint8_t *data, size_t len) {
    uint8_t hash[32];
    size_t hash_size = sizeof(hash);

    tpm2_crypto_hash_accelerated(
        CRYPTO_ALG_SHA256,
        data, len,
        hash, &hash_size
    );
}
```

---

## 📚 Documentation

- **[TPM2-ALGORITHMS.md](docs/TPM2-ALGORITHMS.md)**: Complete algorithm reference
- **[API-GUIDE.md](docs/API-GUIDE.md)**: Detailed API documentation
- **[INTEGRATION.md](docs/INTEGRATION.md)**: DSLLVM integration guide

---

## 🛡️ Security & Compliance

### FIPS 140-2 Approved Algorithms

✅ Hash: SHA-256, SHA-384, SHA-512, SHA3-256/384/512
✅ Symmetric: AES (all modes), TDES
✅ Asymmetric: RSA (2048+), ECC (P-256/384/521)
✅ HMAC: HMAC-SHA256/384/512
✅ KDF: HKDF, PBKDF2, SP800-108

### Hardware Acceleration

All algorithms benefit from hardware acceleration when available:

| Acceleration | Algorithms | Speedup |
|--------------|------------|---------|
| Intel NPU | Hash, HMAC, KDF | 10-50× |
| AES-NI | All AES modes | 4-8× |
| SHA-NI | SHA-256, SHA-512 | 2-4× |
| AVX-512 | Vectorizable ops | 2-16× |

---

## 🔗 Standards Compliance

| Standard | Coverage | Status |
|----------|----------|--------|
| NIST FIPS 186-4 | Digital Signatures | ✅ |
| NIST FIPS 197 | AES | ✅ |
| NIST FIPS 202 | SHA-3 | ✅ |
| NIST SP 800-108 | KDF | ✅ |
| RFC 5869 | HKDF | ✅ |
| RFC 7748 | Curve25519/448 | ✅ |
| RFC 8032 | Ed25519/448 | ✅ |
| GB/T 32918 | SM2/SM3/SM4 | ✅ |

---

## 📝 Version History

- **v2.0.0** (2025-11-25): Integrated into DSLLVM repository
- **v1.0.0** (2025-11-05): Initial implementation (LAT5150DRVMIL)

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Repository**: https://github.com/SWORDIntel/DSLLVM
**Contact**: TPM2 Development Team / DSMIL Kernel Team

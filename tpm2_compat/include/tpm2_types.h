/**
 * @file tpm2_types.h
 * @brief TPM 2.0 Type Definitions and Return Codes
 *
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 * Version: 2.0.0
 * Date: 2025-11-25
 */

#ifndef TPM2_TYPES_H
#define TPM2_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * TPM 2.0 Return Codes
 */
typedef enum {
    TPM2_RC_SUCCESS         = 0x0000,  /**< Operation successful */
    TPM2_RC_FAILURE         = 0x0001,  /**< Generic failure */
    TPM2_RC_SEQUENCE        = 0x0003,  /**< Improper use of sequence */
    TPM2_RC_DISABLED        = 0x0020,  /**< Command is disabled */
    TPM2_RC_MEMORY          = 0x0024,  /**< Out of memory */
    TPM2_RC_HANDLE          = 0x008B,  /**< Invalid handle */
    TPM2_RC_HASH            = 0x0083,  /**< Hash algorithm not supported */
    TPM2_RC_VALUE           = 0x0084,  /**< Invalid value */
    TPM2_RC_SIZE            = 0x0095,  /**< Buffer size error */
    TPM2_RC_SCHEME          = 0x0092,  /**< Unsupported signature scheme */
    TPM2_RC_CURVE           = 0x0093,  /**< Unsupported ECC curve */
    TPM2_RC_SYMMETRIC       = 0x0096,  /**< Unsupported symmetric algorithm */
    TPM2_RC_KEY             = 0x009C,  /**< Invalid key */
    TPM2_RC_SIGNATURE       = 0x009B,  /**< Signature verification failed */
    TPM2_RC_CANCELED        = 0x0909,  /**< Operation canceled */
    TPM2_RC_NOT_SUPPORTED   = 0x0180,  /**< Algorithm/feature not supported */
    TPM2_RC_INSUFFICIENT_BUFFER = 0x01A4  /**< Output buffer too small */
} tpm2_rc_t;

/**
 * TPM 2.0 Cryptographic Algorithm IDs
 */
typedef enum {
    /* Hash Algorithms (10 total) */
    CRYPTO_ALG_NULL         = 0x0000,
    CRYPTO_ALG_SHA1         = 0x0004,
    CRYPTO_ALG_SHA256       = 0x000B,
    CRYPTO_ALG_SHA384       = 0x000C,
    CRYPTO_ALG_SHA512       = 0x000D,
    CRYPTO_ALG_SHA3_256     = 0x0027,
    CRYPTO_ALG_SHA3_384     = 0x0028,
    CRYPTO_ALG_SHA3_512     = 0x0029,
    CRYPTO_ALG_SM3_256      = 0x0012,
    CRYPTO_ALG_SHAKE128     = 0x002A,
    CRYPTO_ALG_SHAKE256     = 0x002B,

    /* Symmetric Encryption - AES (16 modes) */
    CRYPTO_ALG_AES_128_ECB  = 0x1000,
    CRYPTO_ALG_AES_256_ECB  = 0x1001,
    CRYPTO_ALG_AES_128_CBC  = 0x1002,
    CRYPTO_ALG_AES_256_CBC  = 0x1003,
    CRYPTO_ALG_AES_128_CTR  = 0x1004,
    CRYPTO_ALG_AES_256_CTR  = 0x1005,
    CRYPTO_ALG_AES_128_OFB  = 0x1006,
    CRYPTO_ALG_AES_256_OFB  = 0x1007,
    CRYPTO_ALG_AES_128_CFB  = 0x1008,
    CRYPTO_ALG_AES_256_CFB  = 0x1009,
    CRYPTO_ALG_AES_128_GCM  = 0x100A,
    CRYPTO_ALG_AES_256_GCM  = 0x100B,
    CRYPTO_ALG_AES_128_CCM  = 0x100C,
    CRYPTO_ALG_AES_256_CCM  = 0x100D,
    CRYPTO_ALG_AES_128_XTS  = 0x100E,
    CRYPTO_ALG_AES_256_XTS  = 0x100F,

    /* Other Symmetric Ciphers (6 total) */
    CRYPTO_ALG_3DES_EDE     = 0x1010,
    CRYPTO_ALG_CAMELLIA_128 = 0x1011,
    CRYPTO_ALG_CAMELLIA_256 = 0x1012,
    CRYPTO_ALG_SM4_128      = 0x1013,
    CRYPTO_ALG_CHACHA20     = 0x1014,
    CRYPTO_ALG_CHACHA20_POLY1305 = 0x1015,

    /* RSA Key Sizes (5 variants) */
    CRYPTO_ALG_RSA_1024     = 0x2000,
    CRYPTO_ALG_RSA_2048     = 0x2001,
    CRYPTO_ALG_RSA_3072     = 0x2002,
    CRYPTO_ALG_RSA_4096     = 0x2003,
    CRYPTO_ALG_RSA_8192     = 0x2004,

    /* Elliptic Curves (12 curves) */
    CRYPTO_ALG_ECC_P192     = 0x3000,
    CRYPTO_ALG_ECC_P224     = 0x3001,
    CRYPTO_ALG_ECC_P256     = 0x3002,
    CRYPTO_ALG_ECC_P384     = 0x3003,
    CRYPTO_ALG_ECC_P521     = 0x3004,
    CRYPTO_ALG_ECC_SM2_P256 = 0x3005,
    CRYPTO_ALG_ECC_BN_P256  = 0x3006,
    CRYPTO_ALG_ECC_BN_P638  = 0x3007,
    CRYPTO_ALG_ECC_CURVE25519 = 0x3008,
    CRYPTO_ALG_ECC_CURVE448 = 0x3009,
    CRYPTO_ALG_ECC_ED25519  = 0x300A,
    CRYPTO_ALG_ECC_ED448    = 0x300B,

    /* HMAC Algorithms (5 total) */
    CRYPTO_ALG_HMAC_SHA1    = 0x4000,
    CRYPTO_ALG_HMAC_SHA256  = 0x4001,
    CRYPTO_ALG_HMAC_SHA384  = 0x4002,
    CRYPTO_ALG_HMAC_SHA512  = 0x4003,
    CRYPTO_ALG_HMAC_SM3     = 0x4004,

    /* Key Derivation Functions (11 total) */
    CRYPTO_ALG_KDF_SP800_108 = 0x5000,
    CRYPTO_ALG_KDF_SP800_56A = 0x5001,
    CRYPTO_ALG_HKDF_SHA256  = 0x5002,
    CRYPTO_ALG_HKDF_SHA384  = 0x5003,
    CRYPTO_ALG_HKDF_SHA512  = 0x5004,
    CRYPTO_ALG_PBKDF2_SHA256 = 0x5005,
    CRYPTO_ALG_PBKDF2_SHA512 = 0x5006,
    CRYPTO_ALG_SCRYPT       = 0x5007,
    CRYPTO_ALG_ARGON2I      = 0x5008,
    CRYPTO_ALG_ARGON2D      = 0x5009,
    CRYPTO_ALG_ARGON2ID     = 0x500A,

    /* Signature Schemes (8 total) */
    CRYPTO_ALG_RSA_SSA_PKCS1V15 = 0x6000,
    CRYPTO_ALG_RSA_PSS      = 0x6001,
    CRYPTO_ALG_ECDSA_SHA256 = 0x6002,
    CRYPTO_ALG_ECDSA_SHA384 = 0x6003,
    CRYPTO_ALG_ECDSA_SHA512 = 0x6004,
    CRYPTO_ALG_SCHNORR      = 0x6005,
    CRYPTO_ALG_SM2_SIGN     = 0x6006,
    CRYPTO_ALG_ECDAA        = 0x6007,

    /* Key Agreement (3 protocols) */
    CRYPTO_ALG_ECDH         = 0x7000,
    CRYPTO_ALG_ECMQV        = 0x7001,
    CRYPTO_ALG_DH           = 0x7002,

    /* Mask Generation Functions (4 total) */
    CRYPTO_ALG_MGF1_SHA1    = 0x8000,
    CRYPTO_ALG_MGF1_SHA256  = 0x8001,
    CRYPTO_ALG_MGF1_SHA384  = 0x8002,
    CRYPTO_ALG_MGF1_SHA512  = 0x8003,

    /* Post-Quantum Cryptography (8 algorithms) */
    CRYPTO_ALG_KYBER512     = 0x9000,
    CRYPTO_ALG_KYBER768     = 0x9001,
    CRYPTO_ALG_KYBER1024    = 0x9002,
    CRYPTO_ALG_DILITHIUM2   = 0x9003,
    CRYPTO_ALG_DILITHIUM3   = 0x9004,
    CRYPTO_ALG_DILITHIUM5   = 0x9005,
    CRYPTO_ALG_FALCON512    = 0x9006,
    CRYPTO_ALG_FALCON1024   = 0x9007
} tpm2_crypto_algorithm_t;

/**
 * Hardware Acceleration Flags
 */
typedef enum {
    TPM2_ACCEL_NONE         = 0x00,  /**< No hardware acceleration */
    TPM2_ACCEL_NPU          = 0x01,  /**< Intel NPU acceleration */
    TPM2_ACCEL_GNA          = 0x02,  /**< Intel GNA acceleration */
    TPM2_ACCEL_AES_NI       = 0x04,  /**< AES-NI instructions */
    TPM2_ACCEL_SHA_NI       = 0x08,  /**< SHA-NI instructions */
    TPM2_ACCEL_AVX512       = 0x10,  /**< AVX-512 vectorization */
    TPM2_ACCEL_RDRAND       = 0x20,  /**< RDRAND for RNG */
    TPM2_ACCEL_ALL          = 0xFF   /**< All available acceleration */
} tpm2_acceleration_flags_t;

/**
 * Security Level Requirements
 */
typedef enum {
    TPM2_SEC_LEGACY         = 0,  /**< 80-bit security (deprecated) */
    TPM2_SEC_STANDARD       = 1,  /**< 112-bit security */
    TPM2_SEC_HIGH           = 2,  /**< 128-bit security */
    TPM2_SEC_VERY_HIGH      = 3,  /**< 192-bit security */
    TPM2_SEC_ULTRA          = 4   /**< 256-bit security */
} tpm2_security_level_t;

/**
 * Opaque crypto context handle
 */
typedef struct tpm2_crypto_context* tpm2_crypto_context_handle_t;

/**
 * Algorithm capabilities structure
 */
typedef struct {
    tpm2_crypto_algorithm_t algorithm;
    const char *name;
    bool is_supported;
    bool has_hardware_accel;
    uint32_t min_key_size;
    uint32_t max_key_size;
    uint32_t output_size;
} tpm2_algorithm_info_t;

#ifdef __cplusplus
}
#endif

#endif /* TPM2_TYPES_H */

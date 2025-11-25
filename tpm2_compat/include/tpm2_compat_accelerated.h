/**
 * @file tpm2_compat_accelerated.h
 * @brief TPM 2.0 Compatibility Layer with Hardware Acceleration
 *
 * This header provides the main API for TPM 2.0 cryptographic operations
 * with support for hardware acceleration (Intel NPU, AES-NI, SHA-NI, AVX-512).
 *
 * Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
 * Version: 2.0.0
 * Date: 2025-11-25
 */

#ifndef TPM2_COMPAT_ACCELERATED_H
#define TPM2_COMPAT_ACCELERATED_H

#include "tpm2_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup initialization Initialization and Cleanup
 * @{
 */

/**
 * Initialize the TPM2 cryptography subsystem
 *
 * @param accel_flags Hardware acceleration flags to enable
 * @param min_security_level Minimum security level required
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_init(
    tpm2_acceleration_flags_t accel_flags,
    tpm2_security_level_t min_security_level
);

/**
 * Cleanup and release TPM2 cryptography resources
 */
void tpm2_crypto_cleanup(void);

/**
 * Query algorithm capabilities
 *
 * @param algorithm Algorithm to query
 * @param info Pointer to receive algorithm information
 * @return TPM2_RC_SUCCESS on success, TPM2_RC_NOT_SUPPORTED if unavailable
 */
tpm2_rc_t tpm2_crypto_get_algorithm_info(
    tpm2_crypto_algorithm_t algorithm,
    tpm2_algorithm_info_t *info
);

/** @} */

/**
 * @defgroup hash Hash Functions
 * @{
 */

/**
 * Compute cryptographic hash (hardware-accelerated)
 *
 * @param hash_alg Hash algorithm (SHA-256, SHA3-512, etc.)
 * @param data Input data
 * @param data_size Size of input data in bytes
 * @param hash_out Output buffer for hash
 * @param hash_size_inout Input: buffer size, Output: actual hash size
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_hash_accelerated(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *data,
    size_t data_size,
    uint8_t *hash_out,
    size_t *hash_size_inout
);

/** @} */

/**
 * @defgroup symmetric Symmetric Encryption
 * @{
 */

/**
 * Create symmetric encryption/decryption context
 *
 * @param sym_alg Symmetric algorithm (AES-256-GCM, ChaCha20, etc.)
 * @param key Key material
 * @param key_size Key size in bytes
 * @param context_out Pointer to receive context handle
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_context_create(
    tpm2_crypto_algorithm_t sym_alg,
    const uint8_t *key,
    size_t key_size,
    tpm2_crypto_context_handle_t *context_out
);

/**
 * Encrypt data (hardware-accelerated)
 *
 * @param context Crypto context
 * @param plaintext Input plaintext
 * @param plaintext_size Plaintext size in bytes
 * @param iv Initialization vector (mode-specific)
 * @param iv_size IV size in bytes
 * @param ciphertext_out Output buffer for ciphertext
 * @param ciphertext_size_inout Input: buffer size, Output: actual ciphertext size
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_encrypt_accelerated(
    tpm2_crypto_context_handle_t context,
    const uint8_t *plaintext,
    size_t plaintext_size,
    const uint8_t *iv,
    size_t iv_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size_inout
);

/**
 * Decrypt data (hardware-accelerated)
 *
 * @param context Crypto context
 * @param ciphertext Input ciphertext
 * @param ciphertext_size Ciphertext size in bytes
 * @param iv Initialization vector (mode-specific)
 * @param iv_size IV size in bytes
 * @param plaintext_out Output buffer for plaintext
 * @param plaintext_size_inout Input: buffer size, Output: actual plaintext size
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_decrypt_accelerated(
    tpm2_crypto_context_handle_t context,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    const uint8_t *iv,
    size_t iv_size,
    uint8_t *plaintext_out,
    size_t *plaintext_size_inout
);

/**
 * Destroy crypto context
 *
 * @param context Context to destroy
 */
void tpm2_crypto_context_destroy(tpm2_crypto_context_handle_t context);

/** @} */

/**
 * @defgroup aead Authenticated Encryption with Associated Data (AEAD)
 * @{
 */

/**
 * AEAD encryption (GCM, CCM, ChaCha20-Poly1305)
 *
 * @param aead_alg AEAD algorithm
 * @param key Key material
 * @param key_size Key size in bytes
 * @param nonce Nonce/IV
 * @param nonce_size Nonce size in bytes
 * @param aad Additional authenticated data (can be NULL)
 * @param aad_size AAD size in bytes
 * @param plaintext Input plaintext
 * @param plaintext_size Plaintext size
 * @param ciphertext_out Output ciphertext buffer
 * @param ciphertext_size_inout Input: buffer size, Output: actual size
 * @param tag_out Authentication tag output
 * @param tag_size Tag size in bytes
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_aead_encrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *nonce,
    size_t nonce_size,
    const uint8_t *aad,
    size_t aad_size,
    const uint8_t *plaintext,
    size_t plaintext_size,
    uint8_t *ciphertext_out,
    size_t *ciphertext_size_inout,
    uint8_t *tag_out,
    size_t tag_size
);

/**
 * AEAD decryption with authentication
 *
 * @param aead_alg AEAD algorithm
 * @param key Key material
 * @param key_size Key size in bytes
 * @param nonce Nonce/IV
 * @param nonce_size Nonce size in bytes
 * @param aad Additional authenticated data (can be NULL)
 * @param aad_size AAD size in bytes
 * @param ciphertext Input ciphertext
 * @param ciphertext_size Ciphertext size
 * @param tag Authentication tag
 * @param tag_size Tag size in bytes
 * @param plaintext_out Output plaintext buffer
 * @param plaintext_size_inout Input: buffer size, Output: actual size
 * @return TPM2_RC_SUCCESS on success, TPM2_RC_SIGNATURE if auth fails
 */
tpm2_rc_t tpm2_crypto_aead_decrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *nonce,
    size_t nonce_size,
    const uint8_t *aad,
    size_t aad_size,
    const uint8_t *ciphertext,
    size_t ciphertext_size,
    const uint8_t *tag,
    size_t tag_size,
    uint8_t *plaintext_out,
    size_t *plaintext_size_inout
);

/** @} */

/**
 * @defgroup mac Message Authentication Codes (HMAC)
 * @{
 */

/**
 * Compute HMAC (hardware-accelerated)
 *
 * @param hmac_alg HMAC algorithm (HMAC-SHA256, HMAC-SHA512, etc.)
 * @param key Key material
 * @param key_size Key size in bytes
 * @param data Input data
 * @param data_size Data size in bytes
 * @param hmac_out HMAC output buffer
 * @param hmac_size_inout Input: buffer size, Output: actual HMAC size
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_hmac_accelerated(
    tpm2_crypto_algorithm_t hmac_alg,
    const uint8_t *key,
    size_t key_size,
    const uint8_t *data,
    size_t data_size,
    uint8_t *hmac_out,
    size_t *hmac_size_inout
);

/** @} */

/**
 * @defgroup kdf Key Derivation Functions
 * @{
 */

/**
 * HKDF key derivation (RFC 5869)
 *
 * @param hash_alg Hash algorithm for HKDF (SHA-256, SHA-384, SHA-512)
 * @param salt Salt value (can be NULL)
 * @param salt_size Salt size in bytes
 * @param ikm Input key material
 * @param ikm_size IKM size in bytes
 * @param info Application-specific context (can be NULL)
 * @param info_size Info size in bytes
 * @param okm Output key material buffer
 * @param okm_size Desired OKM size in bytes
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_hkdf(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *salt,
    size_t salt_size,
    const uint8_t *ikm,
    size_t ikm_size,
    const uint8_t *info,
    size_t info_size,
    uint8_t *okm,
    size_t okm_size
);

/**
 * PBKDF2 password-based key derivation (RFC 8018)
 *
 * @param hash_alg Hash algorithm for PBKDF2 (SHA-256, SHA-512)
 * @param password Password/passphrase
 * @param password_size Password size in bytes
 * @param salt Salt value
 * @param salt_size Salt size in bytes
 * @param iterations Iteration count (recommended: 100,000+)
 * @param derived_key Output buffer for derived key
 * @param key_size Desired key size in bytes
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_pbkdf2(
    tpm2_crypto_algorithm_t hash_alg,
    const uint8_t *password,
    size_t password_size,
    const uint8_t *salt,
    size_t salt_size,
    uint32_t iterations,
    uint8_t *derived_key,
    size_t key_size
);

/** @} */

/**
 * @defgroup keyagreement Key Agreement Protocols
 * @{
 */

/**
 * Generate ECDH key pair
 *
 * @param curve ECC curve (P-256, P-384, Curve25519, etc.)
 * @param private_key_out Private key output buffer
 * @param private_key_size_inout Input: buffer size, Output: actual size
 * @param public_key_out Public key output buffer
 * @param public_key_size_inout Input: buffer size, Output: actual size
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_ecdh_keygen(
    tpm2_crypto_algorithm_t curve,
    uint8_t *private_key_out,
    size_t *private_key_size_inout,
    uint8_t *public_key_out,
    size_t *public_key_size_inout
);

/**
 * Compute ECDH shared secret
 *
 * @param curve ECC curve
 * @param private_key Our private key
 * @param private_key_size Private key size in bytes
 * @param peer_public_key Peer's public key
 * @param peer_public_key_size Peer public key size in bytes
 * @param shared_secret Shared secret output buffer
 * @param shared_secret_size_inout Input: buffer size, Output: actual size
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_ecdh(
    tpm2_crypto_algorithm_t curve,
    const uint8_t *private_key,
    size_t private_key_size,
    const uint8_t *peer_public_key,
    size_t peer_public_key_size,
    uint8_t *shared_secret,
    size_t *shared_secret_size_inout
);

/** @} */

/**
 * @defgroup utility Utility Functions
 * @{
 */

/**
 * Generate cryptographically secure random bytes
 *
 * @param buffer Output buffer
 * @param size Number of random bytes to generate
 * @return TPM2_RC_SUCCESS on success, error code otherwise
 */
tpm2_rc_t tpm2_crypto_random_bytes(uint8_t *buffer, size_t size);

/**
 * Constant-time memory comparison
 *
 * @param a First buffer
 * @param b Second buffer
 * @param size Size to compare in bytes
 * @return 0 if equal, non-zero if different
 */
int tpm2_crypto_memcmp_constant_time(
    const void *a,
    const void *b,
    size_t size
);

/**
 * Securely zero memory (resistant to compiler optimization)
 *
 * @param buffer Buffer to zero
 * @param size Size in bytes
 */
void tpm2_crypto_secure_zero(void *buffer, size_t size);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* TPM2_COMPAT_ACCELERATED_H */

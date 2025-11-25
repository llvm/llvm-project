#include "tpm2_compat_accelerated.h"
#include <string.h>

tpm2_rc_t tpm2_crypto_aead_encrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key, size_t key_size,
    const uint8_t *nonce, size_t nonce_size,
    const uint8_t *aad, size_t aad_size,
    const uint8_t *plaintext, size_t plaintext_size,
    uint8_t *ciphertext_out, size_t *ciphertext_size_inout,
    uint8_t *tag_out, size_t tag_size)
{
    /* Stub */
    return TPM2_RC_NOT_SUPPORTED;
}

tpm2_rc_t tpm2_crypto_aead_decrypt(
    tpm2_crypto_algorithm_t aead_alg,
    const uint8_t *key, size_t key_size,
    const uint8_t *nonce, size_t nonce_size,
    const uint8_t *aad, size_t aad_size,
    const uint8_t *ciphertext, size_t ciphertext_size,
    const uint8_t *tag, size_t tag_size,
    uint8_t *plaintext_out, size_t *plaintext_size_inout)
{
    /* Stub */
    return TPM2_RC_NOT_SUPPORTED;
}

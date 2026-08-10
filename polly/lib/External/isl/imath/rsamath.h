/*
  Name:     rsamath.h
  Purpose:  Implements part of PKCS#1, v. 2.1, June 14, 2002 (RSA Labs)
  Author:   M. J. Fromberger

  Copyright (C) 2002-2008 Michael J. Fromberger, All Rights Reserved.

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:

  The above copyright notice and this permission notice shall be included in
  all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
 */

#ifndef RSAMATH_H_
#define RSAMATH_H_

#include "imath.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function to fill a buffer with nonzero random bytes */
typedef void (*random_f)(unsigned char *, int);

/* Convert integer to octet string, per PKCS#1 v.2.1 */
mp_result rsa_i2osp(mp_int z, unsigned char *out, int len);

/* Convert octet string to integer, per PKCS#1 v.2.1 */
mp_result rsa_os2ip(mp_int z, unsigned char *in, int len);

/* The following operations assume that you have converted your keys
   and message data into mp_int values somehow.                      */

/* Primitive RSA encryption operation */
mp_result rsa_rsaep(mp_int msg, mp_int exp, mp_int mod, mp_int cipher);

/* Primitive RSA decryption operation */
mp_result rsa_rsadp(mp_int cipher, mp_int exp, mp_int mod, mp_int msg);

/* Primitive RSA signing operation */
mp_result rsa_rsasp(mp_int msg, mp_int exp, mp_int mod, mp_int signature);

/* Primitive RSA verification operation */
mp_result rsa_rsavp(mp_int signature, mp_int exp, mp_int mod, mp_int msg);

/* Compute the maximum length in bytes a message can have using PKCS#1
   v.1.5 encoding with the given modulus */
int       rsa_max_message_len(mp_int mod);

/* Encode a raw message per PKCS#1 v.1.5
   buf      - the buffer containing the message
   msg_len  - the length in bytes of the message
   buf_len  - the size in bytes of the buffer
   tag      - the message tag (nonzero byte)
   filler   - function to generate pseudorandom nonzero padding

   On input, the message is in the first msg_len bytes of the buffer;
   on output, the contents of the buffer are replaced by the padded
   message.  If there is not enough room, MP_RANGE is returned.
 */
mp_result rsa_pkcs1v15_encode(unsigned char *buf, int msg_len, 
			      int buf_len, int tag, random_f filler);

/* Decode a PKCS#1 v.1.5 message back to its raw form 
   buf      - the buffer containing the encoded message
   buf_len  - the length in bytes of the buffer
   tag      - the expected message tag (nonzero byte)
   msg_len  - on output, receives the length of the message content
   
   On output, the message is packed into the first msg_len bytes of
   the buffer, and the rest of the buffer is zeroed.  If the buffer is
   not of the correct form, MP_UNDEF is returned and msg_len is undefined.
 */
mp_result rsa_pkcs1v15_decode(unsigned char *buf, int buf_len, 
			      int tag, int *msg_len);

#ifdef __cplusplus
}
#endif
#endif /* end RSAMATH_H_ */

/*
  Name:     rtest.c
  Purpose:  Test routines for RSA implementation.
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

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rsamath.h"

void random_fill(unsigned char *buf, int len);
void print_buf(unsigned char *buf, int len, int brk, FILE *ofp);

int main(int argc, char *argv[]) {
  int buf_len, msg_len, i;
  unsigned char *buf;
  mp_result res;

  if (argc < 3) {
    fprintf(stderr, "Usage: %s <bufsize> <msglen>\n", argv[0]);
    return 1;
  }

  srand((unsigned int)time(NULL));

  if ((buf_len = atoi(argv[1])) <= 0) {
    fprintf(stderr, "Buffer length must be positive, not %d\n", buf_len);
    return 2;
  }
  if ((msg_len = atoi(argv[2])) <= 0) {
    fprintf(stderr, "Message length must be positive, not %d\n", msg_len);
    return 2;
  }
  if (msg_len > buf_len) msg_len = buf_len;

  buf = calloc(buf_len, sizeof(*buf));
  for (i = 0; i < msg_len; ++i) buf[i] = i + 1;

  printf(
      "Buffer size:  %d bytes\n"
      "Message len:  %d bytes\n\n",
      buf_len, msg_len);

  printf("Message:\n");
  print_buf(buf, msg_len, 16, stdout);
  fputc('\n', stdout);

  if ((res = rsa_pkcs1v15_encode(buf, msg_len, buf_len, 2, random_fill)) !=
      MP_OK) {
    printf("Error from encoding function: %d\n", res);
    free(buf);
    return 1;
  }
  printf("Encoded message:\n");
  print_buf(buf, buf_len, 16, stdout);
  fputc('\n', stdout);

  msg_len = -1; /* make decoder fill this in */
  if ((res = rsa_pkcs1v15_decode(buf, buf_len, 2, &msg_len)) != MP_OK) {
    printf("Error from decoding function: %d\n", res);
    free(buf);
    return 1;
  }
  printf("Decoded message (%d bytes):\n", msg_len);
  print_buf(buf, msg_len, 16, stdout);
  fputc('\n', stdout);

  free(buf);
  return 0;
}

void random_fill(unsigned char *buf, int len) {
  int i;

  for (i = 0; i < len; ++i) {
    unsigned char c = 0;

    while (c == 0) c = (unsigned char)rand();

    buf[i] = c;
  }
}

void print_buf(unsigned char *buf, int len, int brk, FILE *ofp) {
  int i;

  for (i = 0; i < len; ++i) {
    fprintf(ofp, "%02X", buf[i]);

    if ((i + 1) % brk == 0)
      fputc('\n', ofp);
    else
      fputc(' ', ofp);
  }
  if (i % brk) fputc('\n', ofp);
}

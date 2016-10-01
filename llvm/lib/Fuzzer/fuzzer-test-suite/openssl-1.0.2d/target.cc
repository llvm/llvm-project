// Find CVE-2015-3193. Derived from
// https://github.com/hannob/bignum-fuzz/blob/master/CVE-2015-3193-openssl-vs-gcrypt-modexp.c
/* Fuzz-compare the OpenSSL function BN_mod_exp() and the libgcrypt function gcry_mpi_powm().
 *
 * To use this you should compile both libgcrypt and openssl with american fuzzy lop and then statically link everything together, e.g.:
 * afl-clang-fast -o [output] [input] libgcrypt.a libcrypto.a -lgpg-error
 *
 * Input is a binary file, the first bytes will decide how the rest of the file will be split into three bignums.
 *
 * by Hanno Böck, license CC0 (public domain)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <openssl/bn.h>
#include <gcrypt.h>

#define MAXBUF 1000000


struct big_results {
	char *name;
	char *a;
	char *b;
	char *c;
	char *exptmod;
};

void printres(struct big_results *res) {
	printf("\n%s:\n", res->name);
	printf("a: %s\n", res->a);
	printf("b: %s\n", res->b);
	printf("c: %s\n", res->c);
	printf("b^c mod a: %s\n", res->exptmod);
}

void freeres(struct big_results *res) {
	free(res->a);
	free(res->b);
	free(res->c);
	free(res->exptmod);
}


char *gcrytostring(gcry_mpi_t in) {
	char *a, *b;
	size_t i;
	size_t j=0;
	gcry_mpi_aprint(GCRYMPI_FMT_HEX, (unsigned char**) &a, &i, in);
	while(a[j]=='0' && j<(i-2)) j++;
	if ((j%2)==1) j--;
	if (strncmp(&a[j],"00",2)==0) j++;
	b=(char*)malloc(i-j);
	strcpy(b, &a[j]);
	free(a);
	return b;
}

/* test gcry functions from libgcrypt */
void gcrytest(unsigned char* a_raw, int a_len, unsigned char* b_raw, int b_len, unsigned char* c_raw, int c_len, struct big_results *res) {
	gcry_mpi_t a, b, c, res1, res2;

	/* unknown leak here */
	gcry_mpi_scan(&a, GCRYMPI_FMT_USG, a_raw, a_len, NULL);
	res->a = gcrytostring(a);

	gcry_mpi_scan(&b, GCRYMPI_FMT_USG, b_raw, b_len, NULL);
	res->b = gcrytostring(b);

	gcry_mpi_scan(&c, GCRYMPI_FMT_USG, c_raw, c_len, NULL);
	res->c = gcrytostring(c);

	res1=gcry_mpi_new(0);

	gcry_mpi_powm(res1, b, c, a);
	res->exptmod=gcrytostring(res1);

	gcry_mpi_release(a);
	gcry_mpi_release(b);
	gcry_mpi_release(c);
	gcry_mpi_release(res1);
}

/* test bn functions from openssl/libcrypto */
void bntest(unsigned char* a_raw, int a_len, unsigned char* b_raw, int b_len, unsigned char* c_raw, int c_len, struct big_results *res) {
	BN_CTX *bctx = BN_CTX_new();
	BIGNUM *a = BN_new();
	BIGNUM *b = BN_new();
	BIGNUM *c = BN_new();
	BIGNUM *res1 = BN_new();

	BN_bin2bn(a_raw, a_len, a);
	BN_bin2bn(b_raw, b_len, b);
	BN_bin2bn(c_raw, c_len, c);

	res->a = BN_bn2hex(a);
	res->b = BN_bn2hex(b);
	res->c = BN_bn2hex(c);

	BN_mod_exp(res1, b, c, a, bctx);
	res->exptmod = BN_bn2hex(res1);

	BN_free(a);
	BN_free(b);
	BN_free(c);
	BN_free(res1);
	BN_CTX_free(bctx);
}

extern "C" int LLVMFuzzerTestOneInput(const unsigned char *Data, size_t Size) {
	size_t len, l1, l2,l3;
	unsigned int divi1, divi2;
	unsigned char *a, *b, *c;
	struct big_results openssl_results= {"openssl",0,0,0,0};
	struct big_results gcrypt_results= {"libgcrypt",0,0,0,0};

        len = Size;
	if (len<5) return 0;

	divi1=Data[0];
	divi2=Data[1];
	divi1++;divi2++;
	l1 = (len-2)*divi1/256;
	l2 = (len-2-l1)*divi2/256;
	l3 = (len-2-l1-l2);
	assert(l1+l2+l3==len-2);
	//printf("div1 div2 %i %i\n", divi1, divi2);
	//printf("len l1 l2 l3 %i %i %i %i\n", (int)len,(int)l1,(int)l2,(int)l3);
	a=const_cast<unsigned char*>(Data)+2;
	b=const_cast<unsigned char*>(Data)+2+l1;
	c=const_cast<unsigned char*>(Data)+2+l1+l2;


	bntest(a, l1, b, l2, c, l3, &openssl_results);
	//printres(&openssl_results);
	if ((strcmp(openssl_results.a,"0")==0) || (strcmp(openssl_results.c,"0")==0)) goto END;

	gcrytest(a, l1, b, l2, c, l3, &gcrypt_results);
	//printres(&gcrypt_results);

	assert(strcmp(openssl_results.exptmod, gcrypt_results.exptmod)==0);

END:
	freeres(&openssl_results);
	freeres(&gcrypt_results);
	return 0;
}

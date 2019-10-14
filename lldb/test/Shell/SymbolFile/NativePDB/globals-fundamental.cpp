// clang-format off
// REQUIRES: lld

// Test that we can display tag types.
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s 
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/globals-fundamental.lldbinit | FileCheck %s


// Fundamental data types
bool BFalse = false;
// CHECK: (lldb) target variable BFalse
// CHECK-NEXT: (bool) BFalse = false
bool BTrue = true;
// CHECK-NEXT: (lldb) target variable BTrue
// CHECK-NEXT: (bool) BTrue = true
char CA = 'A';
// CHECK-NEXT: (lldb) target variable CA
// CHECK-NEXT: (char) CA = 'A'
char CZ = 'Z';
// CHECK-NEXT: (lldb) target variable CZ
// CHECK-NEXT: (char) CZ = 'Z'
signed char SCa = 'a';
// CHECK-NEXT: (lldb) target variable SCa
// CHECK-NEXT: (signed char) SCa = 'a'
signed char SCz = 'z';
// CHECK-NEXT: (lldb) target variable SCz
// CHECK-NEXT: (signed char) SCz = 'z'
unsigned char UC24 = 24;
// CHECK-NEXT: (lldb) target variable UC24
// CHECK-NEXT: (unsigned char) UC24 = '\x18'
unsigned char UC42 = 42;
// CHECK-NEXT: (lldb) target variable UC42
// CHECK-NEXT: (unsigned char) UC42 = '*'
char16_t C16_24 = u'\24';
// CHECK-NEXT: (lldb) target variable C16_24
// CHECK-NEXT: (char16_t) C16_24 = U+0014
char32_t C32_42 = U'\42';
// CHECK-NEXT: (lldb) target variable C32_42
// CHECK-NEXT: (char32_t) C32_42 = U+0x00000022
wchar_t WC1 = L'1';
// CHECK-NEXT: (lldb) target variable WC1
// CHECK-NEXT: (wchar_t) WC1 = L'1'
wchar_t WCP = L'P';
// CHECK-NEXT: (lldb) target variable WCP
// CHECK-NEXT: (wchar_t) WCP = L'P'
short SMax = 32767;
// CHECK-NEXT: (lldb) target variable SMax
// CHECK-NEXT: (short) SMax = 32767
short SMin = -32768;
// CHECK-NEXT: (lldb) target variable SMin
// CHECK-NEXT: (short) SMin = -32768
unsigned short USMax = 65535;
// CHECK-NEXT: (lldb) target variable USMax
// CHECK-NEXT: (unsigned short) USMax = 65535
unsigned short USMin = 0;
// CHECK-NEXT: (lldb) target variable USMin
// CHECK-NEXT: (unsigned short) USMin = 0
int IMax = 2147483647;
// CHECK-NEXT: (lldb) target variable IMax
// CHECK-NEXT: (int) IMax = 2147483647
int IMin = -2147483648;
// CHECK-NEXT: (lldb) target variable IMin
// CHECK-NEXT: (int) IMin = -2147483648
unsigned int UIMax = 4294967295;
// CHECK-NEXT: (lldb) target variable UIMax
// CHECK-NEXT: (unsigned int) UIMax = 4294967295
unsigned int UIMin = 0;
// CHECK-NEXT: (lldb) target variable UIMin
// CHECK-NEXT: (unsigned int) UIMin = 0
long LMax = 2147483647;
// CHECK-NEXT: (lldb) target variable LMax
// CHECK-NEXT: (long) LMax = 2147483647
long LMin = -2147483648;
// CHECK-NEXT: (lldb) target variable LMin
// CHECK-NEXT: (long) LMin = -2147483648
unsigned long ULMax = 4294967295;
// CHECK-NEXT: (lldb) target variable ULMax
// CHECK-NEXT: (unsigned long) ULMax = 4294967295
unsigned long ULMin = 0;
// CHECK-NEXT: (lldb) target variable ULMin
// CHECK-NEXT: (unsigned long) ULMin = 0
long long LLMax = 9223372036854775807LL;
// CHECK-NEXT: (lldb) target variable LLMax
// CHECK-NEXT: (long long) LLMax = 9223372036854775807
long long LLMin = -9223372036854775807i64 - 1;
// CHECK-NEXT: (lldb) target variable LLMin
// CHECK-NEXT: (long long) LLMin = -9223372036854775808
unsigned long long ULLMax = 18446744073709551615ULL;
// CHECK-NEXT: (lldb) target variable ULLMax
// CHECK-NEXT: (unsigned long long) ULLMax = 18446744073709551615
unsigned long long ULLMin = 0;
// CHECK-NEXT: (lldb) target variable ULLMin
// CHECK-NEXT: (unsigned long long) ULLMin = 0
float F = 3.1415f;
// CHECK-NEXT: (lldb) target variable F
// CHECK-NEXT: (float) F = 3.1415
double D = 3.1415;
// CHECK-NEXT: (lldb) target variable D
// CHECK-NEXT: (double) D = 3.1415

const bool CBFalse = false;
// CHECK-NEXT: (lldb) target variable CBFalse
// CHECK-NEXT: (const bool) CBFalse = false
const bool CBTrue = true;
// CHECK-NEXT: (lldb) target variable CBTrue
// CHECK-NEXT: (const bool) CBTrue = true
const char CCA = 'A';
// CHECK-NEXT: (lldb) target variable CCA
// CHECK-NEXT: (const char) CCA = 'A'
const char CCZ = 'Z';
// CHECK-NEXT: (lldb) target variable CCZ
// CHECK-NEXT: (const char) CCZ = 'Z'
const signed char CSCa = 'a';
// CHECK-NEXT: (lldb) target variable CSCa
// CHECK-NEXT: (const signed char) CSCa = 'a'
const signed char CSCz = 'z';
// CHECK-NEXT: (lldb) target variable CSCz
// CHECK-NEXT: (const signed char) CSCz = 'z'
const unsigned char CUC24 = 24;
// CHECK-NEXT: (lldb) target variable CUC24
// CHECK-NEXT: (const unsigned char) CUC24 = '\x18'
const unsigned char CUC42 = 42;
// CHECK-NEXT: (lldb) target variable CUC42
// CHECK-NEXT: (const unsigned char) CUC42 = '*'
const char16_t CC16_24 = u'\24';
// CHECK-NEXT: (lldb) target variable CC16_24
// CHECK-NEXT: (const char16_t) CC16_24 = U+0014
const char32_t CC32_42 = U'\42';
// CHECK-NEXT: (lldb) target variable CC32_42
// CHECK-NEXT: (const char32_t) CC32_42 = U+0x00000022
const wchar_t CWC1 = L'1';
// CHECK-NEXT: (lldb) target variable CWC1
// CHECK-NEXT: (const wchar_t) CWC1 = L'1'
const wchar_t CWCP = L'P';
// CHECK-NEXT: (lldb) target variable CWCP
// CHECK-NEXT: (const wchar_t) CWCP = L'P'
const short CSMax = 32767;
// CHECK-NEXT: (lldb) target variable CSMax
// CHECK-NEXT: (const short) CSMax = 32767
const short CSMin = -32768;
// CHECK-NEXT: (lldb) target variable CSMin
// CHECK-NEXT: (const short) CSMin = -32768
const unsigned short CUSMax = 65535;
// CHECK-NEXT: (lldb) target variable CUSMax
// CHECK-NEXT: (const unsigned short) CUSMax = 65535
const unsigned short CUSMin = 0;
// CHECK-NEXT: (lldb) target variable CUSMin
// CHECK-NEXT: (const unsigned short) CUSMin = 0
const int CIMax = 2147483647;
// CHECK-NEXT: (lldb) target variable CIMax
// CHECK-NEXT: (const int) CIMax = 2147483647
const int CIMin = -2147483648;
// CHECK-NEXT: (lldb) target variable CIMin
// CHECK-NEXT: (const int) CIMin = -2147483648
const unsigned int CUIMax = 4294967295;
// CHECK-NEXT: (lldb) target variable CUIMax
// CHECK-NEXT: (const unsigned int) CUIMax = 4294967295
const unsigned int CUIMin = 0;
// CHECK-NEXT: (lldb) target variable CUIMin
// CHECK-NEXT: (const unsigned int) CUIMin = 0
const long CLMax = 2147483647;
// CHECK-NEXT: (lldb) target variable CLMax
// CHECK-NEXT: (const long) CLMax = 2147483647
const long CLMin = -2147483648;
// CHECK-NEXT: (lldb) target variable CLMin
// CHECK-NEXT: (const long) CLMin = -2147483648
const unsigned long CULMax = 4294967295;
// CHECK-NEXT: (lldb) target variable CULMax
// CHECK-NEXT: (const unsigned long) CULMax = 4294967295
const unsigned long CULMin = 0;
// CHECK-NEXT: (lldb) target variable CULMin
// CHECK-NEXT: (const unsigned long) CULMin = 0
const long long CLLMax = 9223372036854775807i64;
// CHECK-NEXT: (lldb) target variable CLLMax
// CHECK-NEXT: (const long long) CLLMax = 9223372036854775807
const long long CLLMin = -9223372036854775807i64 - 1;
// CHECK-NEXT: (lldb) target variable CLLMin
// CHECK-NEXT: (const long long) CLLMin = -9223372036854775808
const unsigned long long CULLMax = 18446744073709551615ULL;
// CHECK-NEXT: (lldb) target variable CULLMax
// CHECK-NEXT: (const unsigned long long) CULLMax = 18446744073709551615
const unsigned long long CULLMin = 0;
// CHECK-NEXT: (lldb) target variable CULLMin
// CHECK-NEXT: (const unsigned long long) CULLMin = 0
const float CF = 3.1415f;
// CHECK-NEXT: (lldb) target variable CF
// CHECK-NEXT: (const float) CF = 3.1415
const double CD = 3.1415;
// CHECK-NEXT: (lldb) target variable CD
// CHECK-NEXT: (const double) CD = 3.1415

// constexpr fundamental data types.
constexpr bool ConstexprBFalse = false;
// CHECK-NEXT: (lldb) target variable ConstexprBFalse
// CHECK-NEXT: (const bool) ConstexprBFalse = false
constexpr bool ConstexprBTrue = true;
// CHECK-NEXT: (lldb) target variable ConstexprBTrue
// CHECK-NEXT: (const bool) ConstexprBTrue = true
constexpr char ConstexprCA = 'A';
// CHECK-NEXT: (lldb) target variable ConstexprCA
// CHECK-NEXT: (const char) ConstexprCA = 'A'
constexpr char ConstexprCZ = 'Z';
// CHECK-NEXT: (lldb) target variable ConstexprCZ
// CHECK-NEXT: (const char) ConstexprCZ = 'Z'
constexpr signed char ConstexprSCa = 'a';
// CHECK-NEXT: (lldb) target variable ConstexprSCa
// CHECK-NEXT: (const signed char) ConstexprSCa = 'a'
constexpr signed char ConstexprSCz = 'z';
// CHECK-NEXT: (lldb) target variable ConstexprSCz
// CHECK-NEXT: (const signed char) ConstexprSCz = 'z'
constexpr unsigned char ConstexprUC24 = 24;
// CHECK-NEXT: (lldb) target variable ConstexprUC24
// CHECK-NEXT: (const unsigned char) ConstexprUC24 = '\x18'
constexpr unsigned char ConstexprUC42 = 42;
// CHECK-NEXT: (lldb) target variable ConstexprUC42
// CHECK-NEXT: (const unsigned char) ConstexprUC42 = '*'
constexpr char16_t ConstexprC16_24 = u'\24';
// CHECK-NEXT: (lldb) target variable ConstexprC16_24
// CHECK-NEXT: (const char16_t) ConstexprC16_24 = U+0014
constexpr char32_t ConstexprC32_42 = U'\42';
// CHECK-NEXT: (lldb) target variable ConstexprC32_42
// CHECK-NEXT: (const char32_t) ConstexprC32_42 = U+0x00000022
constexpr wchar_t ConstexprWC1 = L'1';
// CHECK-NEXT: (lldb) target variable ConstexprWC1
// CHECK-NEXT: (const wchar_t) ConstexprWC1 = L'1'
constexpr wchar_t ConstexprWCP = L'P';
// CHECK-NEXT: (lldb) target variable ConstexprWCP
// CHECK-NEXT: (const wchar_t) ConstexprWCP = L'P'
constexpr short ConstexprSMax = 32767;
// CHECK-NEXT: (lldb) target variable ConstexprSMax
// CHECK-NEXT: (const short) ConstexprSMax = 32767
constexpr short ConstexprSMin = -32768;
// CHECK-NEXT: (lldb) target variable ConstexprSMin
// CHECK-NEXT: (const short) ConstexprSMin = -32768
constexpr unsigned short ConstexprUSMax = 65535;
// CHECK-NEXT: (lldb) target variable ConstexprUSMax
// CHECK-NEXT: (const unsigned short) ConstexprUSMax = 65535
constexpr unsigned short ConstexprUSMin = 0;
// CHECK-NEXT: (lldb) target variable ConstexprUSMin
// CHECK-NEXT: (const unsigned short) ConstexprUSMin = 0
constexpr int ConstexprIMax = 2147483647;
// CHECK-NEXT: (lldb) target variable ConstexprIMax
// CHECK-NEXT: (const int) ConstexprIMax = 2147483647
constexpr int ConstexprIMin = -2147483648;
// CHECK-NEXT: (lldb) target variable ConstexprIMin
// CHECK-NEXT: (const int) ConstexprIMin = -2147483648
constexpr unsigned int ConstexprUIMax = 4294967295;
// CHECK-NEXT: (lldb) target variable ConstexprUIMax
// CHECK-NEXT: (const unsigned int) ConstexprUIMax = 4294967295
constexpr unsigned int ConstexprUIMin = 0;
// CHECK-NEXT: (lldb) target variable ConstexprUIMin
// CHECK-NEXT: (const unsigned int) ConstexprUIMin = 0
constexpr long ConstexprLMax = 2147483647;
// CHECK-NEXT: (lldb) target variable ConstexprLMax
// CHECK-NEXT: (const long) ConstexprLMax = 2147483647
constexpr long ConstexprLMin = -2147483648;
// CHECK-NEXT: (lldb) target variable ConstexprLMin
// CHECK-NEXT: (const long) ConstexprLMin = -2147483648
constexpr unsigned long ConstexprULMax = 4294967295;
// CHECK-NEXT: (lldb) target variable ConstexprULMax
// CHECK-NEXT: (const unsigned long) ConstexprULMax = 4294967295
constexpr unsigned long ConstexprULMin = 0;
// CHECK-NEXT: (lldb) target variable ConstexprULMin
// CHECK-NEXT: (const unsigned long) ConstexprULMin = 0
constexpr long long ConstexprLLMax = 9223372036854775807i64;
// CHECK-NEXT: (lldb) target variable ConstexprLLMax
// CHECK-NEXT: (const long long) ConstexprLLMax = 9223372036854775807
constexpr long long ConstexprLLMin = -9223372036854775807i64 - 1;
// CHECK-NEXT: (lldb) target variable ConstexprLLMin
// CHECK-NEXT: (const long long) ConstexprLLMin = -9223372036854775808
constexpr unsigned long long ConstexprULLMax = 18446744073709551615ULL;
// CHECK-NEXT: (lldb) target variable ConstexprULLMax
// CHECK-NEXT: (const unsigned long long) ConstexprULLMax = 18446744073709551615
constexpr unsigned long long ConstexprULLMin = 0;
// CHECK-NEXT: (lldb) target variable ConstexprULLMin
// CHECK-NEXT: (const unsigned long long) ConstexprULLMin = 0
constexpr float ConstexprF = 3.1415f;
// CHECK-NEXT: (lldb) target variable ConstexprF
// CHECK-NEXT: (const float) ConstexprF = 3.1415
constexpr double ConstexprD = 3.1415;
// CHECK-NEXT: (lldb) target variable ConstexprD
// CHECK-NEXT: (const double) ConstexprD = 3.1415


// FIXME: LLDB currently doesn't resolve pointers within the target without a
// running process (I haven't checked whether or not it can with a running
// process). So currently it will just print an address, which is unstable and
// should not be relied upon for testing. So for now we're just checking that
// the variable name and type is correct. We should fix this in LLDB and then
// update the tests.
bool *PBFalse = &BFalse;
// CHECK-NEXT: (lldb) target variable PBFalse
// CHECK-NEXT: (bool *) PBFalse = {{.*}}
bool *PBTrue = &BTrue;
// CHECK-NEXT: (lldb) target variable PBTrue
// CHECK-NEXT: (bool *) PBTrue = {{.*}}
char *PCA = &CA;
// CHECK-NEXT: (lldb) target variable PCA
// CHECK-NEXT: (char *) PCA = {{.*}}
char *PCZ = &CZ;
// CHECK-NEXT: (lldb) target variable PCZ
// CHECK-NEXT: (char *) PCZ = {{.*}}
signed char *PSCa = &SCa;
// CHECK-NEXT: (lldb) target variable PSCa
// CHECK-NEXT: (signed char *) PSCa = {{.*}}
signed char *PSCz = &SCz;
// CHECK-NEXT: (lldb) target variable PSCz
// CHECK-NEXT: (signed char *) PSCz = {{.*}}
unsigned char *PUC24 = &UC24;
// CHECK-NEXT: (lldb) target variable PUC24
// CHECK-NEXT: (unsigned char *) PUC24 = {{.*}}
unsigned char *PUC42 = &UC42;
// CHECK-NEXT: (lldb) target variable PUC42
// CHECK-NEXT: (unsigned char *) PUC42 = {{.*}}
char16_t *PC16_24 = &C16_24;
// CHECK-NEXT: (lldb) target variable PC16_24
// CHECK-NEXT: (char16_t *) PC16_24 = {{.*}}
char32_t *PC32_42 = &C32_42;
// CHECK-NEXT: (lldb) target variable PC32_42
// CHECK-NEXT: (char32_t *) PC32_42 = {{.*}}
wchar_t *PWC1 = &WC1;
// CHECK-NEXT: (lldb) target variable PWC1
// CHECK-NEXT: (wchar_t *) PWC1 = {{.*}}
wchar_t *PWCP = &WCP;
// CHECK-NEXT: (lldb) target variable PWCP
// CHECK-NEXT: (wchar_t *) PWCP = {{.*}}
short *PSMax = &SMax;
// CHECK-NEXT: (lldb) target variable PSMax
// CHECK-NEXT: (short *) PSMax = {{.*}}
short *PSMin = &SMin;
// CHECK-NEXT: (lldb) target variable PSMin
// CHECK-NEXT: (short *) PSMin = {{.*}}
unsigned short *PUSMax = &USMax;
// CHECK-NEXT: (lldb) target variable PUSMax
// CHECK-NEXT: (unsigned short *) PUSMax = {{.*}}
unsigned short *PUSMin = &USMin;
// CHECK-NEXT: (lldb) target variable PUSMin
// CHECK-NEXT: (unsigned short *) PUSMin = {{.*}}
int *PIMax = &IMax;
// CHECK-NEXT: (lldb) target variable PIMax
// CHECK-NEXT: (int *) PIMax = {{.*}}
int *PIMin = &IMin;
// CHECK-NEXT: (lldb) target variable PIMin
// CHECK-NEXT: (int *) PIMin = {{.*}}
unsigned int *PUIMax = &UIMax;
// CHECK-NEXT: (lldb) target variable PUIMax
// CHECK-NEXT: (unsigned int *) PUIMax = {{.*}}
unsigned int *PUIMin = &UIMin;
// CHECK-NEXT: (lldb) target variable PUIMin
// CHECK-NEXT: (unsigned int *) PUIMin = {{.*}}
long *PLMax = &LMax;
// CHECK-NEXT: (lldb) target variable PLMax
// CHECK-NEXT: (long *) PLMax = {{.*}}
long *PLMin = &LMin;
// CHECK-NEXT: (lldb) target variable PLMin
// CHECK-NEXT: (long *) PLMin = {{.*}}
unsigned long *PULMax = &ULMax;
// CHECK-NEXT: (lldb) target variable PULMax
// CHECK-NEXT: (unsigned long *) PULMax = {{.*}}
unsigned long *PULMin = &ULMin;
// CHECK-NEXT: (lldb) target variable PULMin
// CHECK-NEXT: (unsigned long *) PULMin = {{.*}}
long long *PLLMax = &LLMax;
// CHECK-NEXT: (lldb) target variable PLLMax
// CHECK-NEXT: (long long *) PLLMax = {{.*}}
long long *PLLMin = &LLMin;
// CHECK-NEXT: (lldb) target variable PLLMin
// CHECK-NEXT: (long long *) PLLMin = {{.*}}
unsigned long long *PULLMax = &ULLMax;
// CHECK-NEXT: (lldb) target variable PULLMax
// CHECK-NEXT: (unsigned long long *) PULLMax = {{.*}}
unsigned long long *PULLMin = &ULLMin;
// CHECK-NEXT: (lldb) target variable PULLMin
// CHECK-NEXT: (unsigned long long *) PULLMin = {{.*}}
float *PF = &F;
// CHECK-NEXT: (lldb) target variable PF
// CHECK-NEXT: (float *) PF = {{.*}}
double *PD = &D;
// CHECK-NEXT: (lldb) target variable PD
// CHECK-NEXT: (double *) PD = {{.*}}

// Const pointers to fundamental data types
const bool *CPBFalse = &BFalse;
// CHECK-NEXT: (lldb) target variable CPBFalse
// CHECK-NEXT: (const bool *) CPBFalse = {{.*}}
const bool *CPBTrue = &BTrue;
// CHECK-NEXT: (lldb) target variable CPBTrue
// CHECK-NEXT: (const bool *) CPBTrue = {{.*}}
const char *CPCA = &CA;
// CHECK-NEXT: (lldb) target variable CPCA
// CHECK-NEXT: (const char *) CPCA = {{.*}}
const char *CPCZ = &CZ;
// CHECK-NEXT: (lldb) target variable CPCZ
// CHECK-NEXT: (const char *) CPCZ = {{.*}}
const signed char *CPSCa = &SCa;
// CHECK-NEXT: (lldb) target variable CPSCa
// CHECK-NEXT: (const signed char *) CPSCa = {{.*}}
const signed char *CPSCz = &SCz;
// CHECK-NEXT: (lldb) target variable CPSCz
// CHECK-NEXT: (const signed char *) CPSCz = {{.*}}
const unsigned char *CPUC24 = &UC24;
// CHECK-NEXT: (lldb) target variable CPUC24
// CHECK-NEXT: (const unsigned char *) CPUC24 = {{.*}}
const unsigned char *CPUC42 = &UC42;
// CHECK-NEXT: (lldb) target variable CPUC42
// CHECK-NEXT: (const unsigned char *) CPUC42 = {{.*}}
const char16_t *CPC16_24 = &C16_24;
// CHECK-NEXT: (lldb) target variable CPC16_24
// CHECK-NEXT: (const char16_t *) CPC16_24 = {{.*}}
const char32_t *CPC32_42 = &C32_42;
// CHECK-NEXT: (lldb) target variable CPC32_42
// CHECK-NEXT: (const char32_t *) CPC32_42 = {{.*}}
const wchar_t *CPWC1 = &WC1;
// CHECK-NEXT: (lldb) target variable CPWC1
// CHECK-NEXT: (const wchar_t *) CPWC1 = {{.*}}
const wchar_t *CPWCP = &WCP;
// CHECK-NEXT: (lldb) target variable CPWCP
// CHECK-NEXT: (const wchar_t *) CPWCP = {{.*}}
const short *CPSMax = &SMax;
// CHECK-NEXT: (lldb) target variable CPSMax
// CHECK-NEXT: (const short *) CPSMax = {{.*}}
const short *CPSMin = &SMin;
// CHECK-NEXT: (lldb) target variable CPSMin
// CHECK-NEXT: (const short *) CPSMin = {{.*}}
const unsigned short *CPUSMax = &USMax;
// CHECK-NEXT: (lldb) target variable CPUSMax
// CHECK-NEXT: (const unsigned short *) CPUSMax = {{.*}}
const unsigned short *CPUSMin = &USMin;
// CHECK-NEXT: (lldb) target variable CPUSMin
// CHECK-NEXT: (const unsigned short *) CPUSMin = {{.*}}
const int *CPIMax = &IMax;
// CHECK-NEXT: (lldb) target variable CPIMax
// CHECK-NEXT: (const int *) CPIMax = {{.*}}
const int *CPIMin = &IMin;
// CHECK-NEXT: (lldb) target variable CPIMin
// CHECK-NEXT: (const int *) CPIMin = {{.*}}
const unsigned int *CPUIMax = &UIMax;
// CHECK-NEXT: (lldb) target variable CPUIMax
// CHECK-NEXT: (const unsigned int *) CPUIMax = {{.*}}
const unsigned int *CPUIMin = &UIMin;
// CHECK-NEXT: (lldb) target variable CPUIMin
// CHECK-NEXT: (const unsigned int *) CPUIMin = {{.*}}
const long *CPLMax = &LMax;
// CHECK-NEXT: (lldb) target variable CPLMax
// CHECK-NEXT: (const long *) CPLMax = {{.*}}
const long *CPLMin = &LMin;
// CHECK-NEXT: (lldb) target variable CPLMin
// CHECK-NEXT: (const long *) CPLMin = {{.*}}
const unsigned long *CPULMax = &ULMax;
// CHECK-NEXT: (lldb) target variable CPULMax
// CHECK-NEXT: (const unsigned long *) CPULMax = {{.*}}
const unsigned long *CPULMin = &ULMin;
// CHECK-NEXT: (lldb) target variable CPULMin
// CHECK-NEXT: (const unsigned long *) CPULMin = {{.*}}
const long long *CPLLMax = &LLMax;
// CHECK-NEXT: (lldb) target variable CPLLMax
// CHECK-NEXT: (const long long *) CPLLMax = {{.*}}
const long long *CPLLMin = &LLMin;
// CHECK-NEXT: (lldb) target variable CPLLMin
// CHECK-NEXT: (const long long *) CPLLMin = {{.*}}
const unsigned long long *CPULLMax = &ULLMax;
// CHECK-NEXT: (lldb) target variable CPULLMax
// CHECK-NEXT: (const unsigned long long *) CPULLMax = {{.*}}
const unsigned long long *CPULLMin = &ULLMin;
// CHECK-NEXT: (lldb) target variable CPULLMin
// CHECK-NEXT: (const unsigned long long *) CPULLMin = {{.*}}
const float *CPF = &F;
// CHECK-NEXT: (lldb) target variable CPF
// CHECK-NEXT: (const float *) CPF = {{.*}}
const double *CPD = &D;
// CHECK-NEXT: (lldb) target variable CPD
// CHECK-NEXT: (const double *) CPD = {{.*}}


// References to fundamental data types

bool &RBFalse = BFalse;
// CHECK-NEXT: (lldb) target variable RBFalse
// CHECK-NEXT: (bool &) RBFalse = {{.*}} (&::RBFalse = false)
bool &RBTrue = BTrue;
// CHECK-NEXT: (lldb) target variable RBTrue
// CHECK-NEXT: (bool &) RBTrue = {{.*}} (&::RBTrue = true)
char &RCA = CA;
// CHECK-NEXT: (lldb) target variable RCA
// CHECK-NEXT: (char &) RCA = {{.*}} (&::RCA = 'A')
char &RCZ = CZ;
// CHECK-NEXT: (lldb) target variable RCZ
// CHECK-NEXT: (char &) RCZ = {{.*}} (&::RCZ = 'Z')
signed char &RSCa = SCa;
// CHECK-NEXT: (lldb) target variable RSCa
// CHECK-NEXT: (signed char &) RSCa = {{.*}} (&::RSCa = 'a')
signed char &RSCz = SCz;
// CHECK-NEXT: (lldb) target variable RSCz
// CHECK-NEXT: (signed char &) RSCz = {{.*}} (&::RSCz = 'z')
unsigned char &RUC24 = UC24;
// CHECK-NEXT: (lldb) target variable RUC24
// CHECK-NEXT: (unsigned char &) RUC24 = {{.*}} (&::RUC24 = '\x18')
unsigned char &RUC42 = UC42;
// CHECK-NEXT: (lldb) target variable RUC42
// CHECK-NEXT: (unsigned char &) RUC42 = {{.*}} (&::RUC42 = '*')
short &RSMax = SMax;
// CHECK-NEXT: (lldb) target variable RSMax
// CHECK-NEXT: (short &) RSMax = {{.*}} (&::RSMax = 32767)
short &RSMin = SMin;
// CHECK-NEXT: (lldb) target variable RSMin
// CHECK-NEXT: (short &) RSMin = {{.*}} (&::RSMin = -32768)
unsigned short &RUSMax = USMax;
// CHECK-NEXT: (lldb) target variable RUSMax
// CHECK-NEXT: (unsigned short &) RUSMax = {{.*}} (&::RUSMax = 65535)
unsigned short &RUSMin = USMin;
// CHECK-NEXT: (lldb) target variable RUSMin
// CHECK-NEXT: (unsigned short &) RUSMin = {{.*}} (&::RUSMin = 0)
int &RIMax = IMax;
// CHECK-NEXT: (lldb) target variable RIMax
// CHECK-NEXT: (int &) RIMax = {{.*}} (&::RIMax = 2147483647)
int &RIMin = IMin;
// CHECK-NEXT: (lldb) target variable RIMin
// CHECK-NEXT: (int &) RIMin = {{.*}} (&::RIMin = -2147483648)
unsigned int &RUIMax = UIMax;
// CHECK-NEXT: (lldb) target variable RUIMax
// CHECK-NEXT: (unsigned int &) RUIMax = {{.*}} (&::RUIMax = 4294967295)
unsigned int &RUIMin = UIMin;
// CHECK-NEXT: (lldb) target variable RUIMin
// CHECK-NEXT: (unsigned int &) RUIMin = {{.*}} (&::RUIMin = 0)
long &RLMax = LMax;
// CHECK-NEXT: (lldb) target variable RLMax
// CHECK-NEXT: (long &) RLMax = {{.*}} (&::RLMax = 2147483647)
long &RLMin = LMin;
// CHECK-NEXT: (lldb) target variable RLMin
// CHECK-NEXT: (long &) RLMin = {{.*}} (&::RLMin = -2147483648)
unsigned long &RULMax = ULMax;
// CHECK-NEXT: (lldb) target variable RULMax
// CHECK-NEXT: (unsigned long &) RULMax = {{.*}} (&::RULMax = 4294967295)
unsigned long &RULMin = ULMin;
// CHECK-NEXT: (lldb) target variable RULMin
// CHECK-NEXT: (unsigned long &) RULMin = {{.*}} (&::RULMin = 0)
long long &RLLMax = LLMax;
// CHECK-NEXT: (lldb) target variable RLLMax
// CHECK-NEXT: (long long &) RLLMax = {{.*}} (&::RLLMax = 9223372036854775807)
long long &RLLMin = LLMin;
// CHECK-NEXT: (lldb) target variable RLLMin
// CHECK-NEXT: (long long &) RLLMin = {{.*}} (&::RLLMin = -9223372036854775808)
unsigned long long &RULLMax = ULLMax;
// CHECK-NEXT: (lldb) target variable RULLMax
// CHECK-NEXT: (unsigned long long &) RULLMax = {{.*}} (&::RULLMax = 18446744073709551615)
unsigned long long &RULLMin = ULLMin;
// CHECK-NEXT: (lldb) target variable RULLMin
// CHECK-NEXT: (unsigned long long &) RULLMin = {{.*}} (&::RULLMin = 0)
float &RF = F;
// CHECK-NEXT: (lldb) target variable RF
// CHECK-NEXT: (float &) RF = {{.*}} (&::RF = 3.1415)
double &RD = D;
// CHECK-NEXT: (lldb) target variable RD
// CHECK-NEXT: (double &) RD = {{.*}} (&::RD = 3.1415000000000002)

// const references to fundamental data types
const bool &CRBFalse = BFalse;
// CHECK-NEXT: (lldb) target variable CRBFalse
// CHECK-NEXT: (const bool &) CRBFalse = {{.*}} (&::CRBFalse = false)
const bool &CRBTrue = BTrue;
// CHECK-NEXT: (lldb) target variable CRBTrue
// CHECK-NEXT: (const bool &) CRBTrue = {{.*}} (&::CRBTrue = true)
const char &CRCA = CA;
// CHECK-NEXT: (lldb) target variable CRCA
// CHECK-NEXT: (const char &) CRCA = {{.*}} (&::CRCA = 'A')
const char &CRCZ = CZ;
// CHECK-NEXT: (lldb) target variable CRCZ
// CHECK-NEXT: (const char &) CRCZ = {{.*}} (&::CRCZ = 'Z')
const signed char &CRSCa = SCa;
// CHECK-NEXT: (lldb) target variable CRSCa
// CHECK-NEXT: (const signed char &) CRSCa = {{.*}} (&::CRSCa = 'a')
const signed char &CRSCz = SCz;
// CHECK-NEXT: (lldb) target variable CRSCz
// CHECK-NEXT: (const signed char &) CRSCz = {{.*}} (&::CRSCz = 'z')
const unsigned char &CRUC24 = UC24;
// CHECK-NEXT: (lldb) target variable CRUC24
// CHECK-NEXT: (const unsigned char &) CRUC24 = {{.*}} (&::CRUC24 = '\x18')
const unsigned char &CRUC42 = UC42;
// CHECK-NEXT: (lldb) target variable CRUC42
// CHECK-NEXT: (const unsigned char &) CRUC42 = {{.*}} (&::CRUC42 = '*')
const short &CRSMax = SMax;
// CHECK-NEXT: (lldb) target variable CRSMax
// CHECK-NEXT: (const short &) CRSMax = {{.*}} (&::CRSMax = 32767)
const short &CRSMin = SMin;
// CHECK-NEXT: (lldb) target variable CRSMin
// CHECK-NEXT: (const short &) CRSMin = {{.*}} (&::CRSMin = -32768)
const unsigned short &CRUSMax = USMax;
// CHECK-NEXT: (lldb) target variable CRUSMax
// CHECK-NEXT: (const unsigned short &) CRUSMax = {{.*}} (&::CRUSMax = 65535)
const unsigned short &CRUSMin = USMin;
// CHECK-NEXT: (lldb) target variable CRUSMin
// CHECK-NEXT: (const unsigned short &) CRUSMin = {{.*}} (&::CRUSMin = 0)
const int &CRIMax = IMax;
// CHECK-NEXT: (lldb) target variable CRIMax
// CHECK-NEXT: (const int &) CRIMax = {{.*}} (&::CRIMax = 2147483647)
const int &CRIMin = IMin;
// CHECK-NEXT: (lldb) target variable CRIMin
// CHECK-NEXT: (const int &) CRIMin = {{.*}} (&::CRIMin = -2147483648)
const unsigned int &CRUIMax = UIMax;
// CHECK-NEXT: (lldb) target variable CRUIMax
// CHECK-NEXT: (const unsigned int &) CRUIMax = {{.*}} (&::CRUIMax = 4294967295)
const unsigned int &CRUIMin = UIMin;
// CHECK-NEXT: (lldb) target variable CRUIMin
// CHECK-NEXT: (const unsigned int &) CRUIMin = {{.*}} (&::CRUIMin = 0)
const long &CRLMax = LMax;
// CHECK-NEXT: (lldb) target variable CRLMax
// CHECK-NEXT: (const long &) CRLMax = {{.*}} (&::CRLMax = 2147483647)
const long &CRLMin = LMin;
// CHECK-NEXT: (lldb) target variable CRLMin
// CHECK-NEXT: (const long &) CRLMin = {{.*}} (&::CRLMin = -2147483648)
const unsigned long &CRULMax = ULMax;
// CHECK-NEXT: (lldb) target variable CRULMax
// CHECK-NEXT: (const unsigned long &) CRULMax = {{.*}} (&::CRULMax = 4294967295)
const unsigned long &CRULMin = ULMin;
// CHECK-NEXT: (lldb) target variable CRULMin
// CHECK-NEXT: (const unsigned long &) CRULMin = {{.*}} (&::CRULMin = 0)
const long long &CRLLMax = LLMax;
// CHECK-NEXT: (lldb) target variable CRLLMax
// CHECK-NEXT: (const long long &) CRLLMax = {{.*}} (&::CRLLMax = 9223372036854775807)
const long long &CRLLMin = LLMin;
// CHECK-NEXT: (lldb) target variable CRLLMin
// CHECK-NEXT: (const long long &) CRLLMin = {{.*}} (&::CRLLMin = -9223372036854775808)
const unsigned long long &CRULLMax = ULLMax;
// CHECK-NEXT: (lldb) target variable CRULLMax
// CHECK-NEXT: (const unsigned long long &) CRULLMax = {{.*}} (&::CRULLMax = 18446744073709551615)
const unsigned long long &CRULLMin = ULLMin;
// CHECK-NEXT: (lldb) target variable CRULLMin
// CHECK-NEXT: (const unsigned long long &) CRULLMin = {{.*}} (&::CRULLMin = 0)
const float &CRF = F;
// CHECK-NEXT: (lldb) target variable CRF
// CHECK-NEXT: (const float &) CRF = {{.*}} (&::CRF = 3.1415)
const double &CRD = D;
// CHECK-NEXT: (lldb) target variable CRD
// CHECK-NEXT: (const double &) CRD = {{.*}} (&::CRD = 3.1415000000000002)

char16_t &RC16_24 = C16_24;
// CHECK: (lldb) target variable RC16_24
// FIXME: (char16_t &) RC16_24 = {{.*}} (&::RC16_24 = U+0014)
char32_t &RC32_42 = C32_42;
// CHECK: (lldb) target variable RC32_42
// FIXME: (char32_t &) RC32_42 = {{.*}} (&::RC32_42 = U+0x00000022)
wchar_t &RWC1 = WC1;
// CHECK: (lldb) target variable RWC1
// FIXME: (wchar_t &) RWC1 = {{.*}} (&::RWC1 = L'1')
wchar_t &RWCP = WCP;
// CHECK: (lldb) target variable RWCP
// FIXME: (wchar_t &) RWCP = {{.*}} (&::RWCP = L'P')
const char16_t &CRC16_24 = C16_24;
// CHECK: (lldb) target variable CRC16_24
// FIXME: (const char16_t &) CRC16_24 = {{.*}} (&::CRC16_24 = U+0014)
const char32_t &CRC32_42 = C32_42;
// CHECK: (lldb) target variable CRC32_42
// FIXME: (const char32_t &) CRC32_42 = {{.*}} (&::CRC32_42 = U+0x00000022)
const wchar_t &CRWC1 = WC1;
// CHECK: (lldb) target variable CRWC1
// FIXME: (const wchar_t &) CRWC1 = {{.*}} (&::CRWC1 = L'1')
const wchar_t &CRWCP = WCP;
// CHECK: (lldb) target variable CRWCP
// FIXME: (const wchar_t &) CRWCP = {{.*}} (&::CRWCP = L'P')


// CHECK:      TranslationUnitDecl {{.*}}
// CHECK-NEXT: |-VarDecl {{.*}} BFalse 'bool'
// CHECK-NEXT: |-VarDecl {{.*}} BTrue 'bool'
// CHECK-NEXT: |-VarDecl {{.*}} CA 'char'
// CHECK-NEXT: |-VarDecl {{.*}} CZ 'char'
// CHECK-NEXT: |-VarDecl {{.*}} SCa 'signed char'
// CHECK-NEXT: |-VarDecl {{.*}} SCz 'signed char'
// CHECK-NEXT: |-VarDecl {{.*}} UC24 'unsigned char'
// CHECK-NEXT: |-VarDecl {{.*}} UC42 'unsigned char'
// CHECK-NEXT: |-VarDecl {{.*}} C16_24 'char16_t'
// CHECK-NEXT: |-VarDecl {{.*}} C32_42 'char32_t'
// CHECK-NEXT: |-VarDecl {{.*}} WC1 'wchar_t'
// CHECK-NEXT: |-VarDecl {{.*}} WCP 'wchar_t'
// CHECK-NEXT: |-VarDecl {{.*}} SMax 'short'
// CHECK-NEXT: |-VarDecl {{.*}} SMin 'short'
// CHECK-NEXT: |-VarDecl {{.*}} USMax 'unsigned short'
// CHECK-NEXT: |-VarDecl {{.*}} USMin 'unsigned short'
// CHECK-NEXT: |-VarDecl {{.*}} IMax 'int'
// CHECK-NEXT: |-VarDecl {{.*}} IMin 'int'
// CHECK-NEXT: |-VarDecl {{.*}} UIMax 'unsigned int'
// CHECK-NEXT: |-VarDecl {{.*}} UIMin 'unsigned int'
// CHECK-NEXT: |-VarDecl {{.*}} LMax 'long'
// CHECK-NEXT: |-VarDecl {{.*}} LMin 'long'
// CHECK-NEXT: |-VarDecl {{.*}} ULMax 'unsigned long'
// CHECK-NEXT: |-VarDecl {{.*}} ULMin 'unsigned long'
// CHECK-NEXT: |-VarDecl {{.*}} LLMax 'long long'
// CHECK-NEXT: |-VarDecl {{.*}} LLMin 'long long'
// CHECK-NEXT: |-VarDecl {{.*}} ULLMax 'unsigned long long'
// CHECK-NEXT: |-VarDecl {{.*}} ULLMin 'unsigned long long'
// CHECK-NEXT: |-VarDecl {{.*}} F 'float'
// CHECK-NEXT: |-VarDecl {{.*}} D 'double'
// CHECK-NEXT: |-VarDecl {{.*}} CBFalse 'const bool'
// CHECK-NEXT: |-VarDecl {{.*}} CBTrue 'const bool'
// CHECK-NEXT: |-VarDecl {{.*}} CCA 'const char'
// CHECK-NEXT: |-VarDecl {{.*}} CCZ 'const char'
// CHECK-NEXT: |-VarDecl {{.*}} CSCa 'const signed char'
// CHECK-NEXT: |-VarDecl {{.*}} CSCz 'const signed char'
// CHECK-NEXT: |-VarDecl {{.*}} CUC24 'const unsigned char'
// CHECK-NEXT: |-VarDecl {{.*}} CUC42 'const unsigned char'
// CHECK-NEXT: |-VarDecl {{.*}} CC16_24 'const char16_t'
// CHECK-NEXT: |-VarDecl {{.*}} CC32_42 'const char32_t'
// CHECK-NEXT: |-VarDecl {{.*}} CWC1 'const wchar_t'
// CHECK-NEXT: |-VarDecl {{.*}} CWCP 'const wchar_t'
// CHECK-NEXT: |-VarDecl {{.*}} CSMax 'const short'
// CHECK-NEXT: |-VarDecl {{.*}} CSMin 'const short'
// CHECK-NEXT: |-VarDecl {{.*}} CUSMax 'const unsigned short'
// CHECK-NEXT: |-VarDecl {{.*}} CUSMin 'const unsigned short'
// CHECK-NEXT: |-VarDecl {{.*}} CIMax 'const int'
// CHECK-NEXT: |-VarDecl {{.*}} CIMin 'const int'
// CHECK-NEXT: |-VarDecl {{.*}} CUIMax 'const unsigned int'
// CHECK-NEXT: |-VarDecl {{.*}} CUIMin 'const unsigned int'
// CHECK-NEXT: |-VarDecl {{.*}} CLMax 'const long'
// CHECK-NEXT: |-VarDecl {{.*}} CLMin 'const long'
// CHECK-NEXT: |-VarDecl {{.*}} CULMax 'const unsigned long'
// CHECK-NEXT: |-VarDecl {{.*}} CULMin 'const unsigned long'
// CHECK-NEXT: |-VarDecl {{.*}} CLLMax 'const long long'
// CHECK-NEXT: |-VarDecl {{.*}} CLLMin 'const long long'
// CHECK-NEXT: |-VarDecl {{.*}} CULLMax 'const unsigned long long'
// CHECK-NEXT: |-VarDecl {{.*}} CULLMin 'const unsigned long long'
// CHECK-NEXT: |-VarDecl {{.*}} CF 'const float'
// CHECK-NEXT: |-VarDecl {{.*}} CD 'const double'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprBFalse 'const bool'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprBTrue 'const bool'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprCA 'const char'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprCZ 'const char'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprSCa 'const signed char'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprSCz 'const signed char'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprUC24 'const unsigned char'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprUC42 'const unsigned char'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprC16_24 'const char16_t'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprC32_42 'const char32_t'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprWC1 'const wchar_t'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprWCP 'const wchar_t'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprSMax 'const short'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprSMin 'const short'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprUSMax 'const unsigned short'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprUSMin 'const unsigned short'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprIMax 'const int'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprIMin 'const int'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprUIMax 'const unsigned int'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprUIMin 'const unsigned int'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprLMax 'const long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprLMin 'const long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprULMax 'const unsigned long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprULMin 'const unsigned long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprLLMax 'const long long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprLLMin 'const long long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprULLMax 'const unsigned long long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprULLMin 'const unsigned long long'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprF 'const float'
// CHECK-NEXT: |-VarDecl {{.*}} ConstexprD 'const double'
// CHECK-NEXT: |-VarDecl {{.*}} PBFalse 'bool *'
// CHECK-NEXT: |-VarDecl {{.*}} PBTrue 'bool *'
// CHECK-NEXT: |-VarDecl {{.*}} PCA 'char *'
// CHECK-NEXT: |-VarDecl {{.*}} PCZ 'char *'
// CHECK-NEXT: |-VarDecl {{.*}} PSCa 'signed char *'
// CHECK-NEXT: |-VarDecl {{.*}} PSCz 'signed char *'
// CHECK-NEXT: |-VarDecl {{.*}} PUC24 'unsigned char *'
// CHECK-NEXT: |-VarDecl {{.*}} PUC42 'unsigned char *'
// CHECK-NEXT: |-VarDecl {{.*}} PC16_24 'char16_t *'
// CHECK-NEXT: |-VarDecl {{.*}} PC32_42 'char32_t *'
// CHECK-NEXT: |-VarDecl {{.*}} PWC1 'wchar_t *'
// CHECK-NEXT: |-VarDecl {{.*}} PWCP 'wchar_t *'
// CHECK-NEXT: |-VarDecl {{.*}} PSMax 'short *'
// CHECK-NEXT: |-VarDecl {{.*}} PSMin 'short *'
// CHECK-NEXT: |-VarDecl {{.*}} PUSMax 'unsigned short *'
// CHECK-NEXT: |-VarDecl {{.*}} PUSMin 'unsigned short *'
// CHECK-NEXT: |-VarDecl {{.*}} PIMax 'int *'
// CHECK-NEXT: |-VarDecl {{.*}} PIMin 'int *'
// CHECK-NEXT: |-VarDecl {{.*}} PUIMax 'unsigned int *'
// CHECK-NEXT: |-VarDecl {{.*}} PUIMin 'unsigned int *'
// CHECK-NEXT: |-VarDecl {{.*}} PLMax 'long *'
// CHECK-NEXT: |-VarDecl {{.*}} PLMin 'long *'
// CHECK-NEXT: |-VarDecl {{.*}} PULMax 'unsigned long *'
// CHECK-NEXT: |-VarDecl {{.*}} PULMin 'unsigned long *'
// CHECK-NEXT: |-VarDecl {{.*}} PLLMax 'long long *'
// CHECK-NEXT: |-VarDecl {{.*}} PLLMin 'long long *'
// CHECK-NEXT: |-VarDecl {{.*}} PULLMax 'unsigned long long *'
// CHECK-NEXT: |-VarDecl {{.*}} PULLMin 'unsigned long long *'
// CHECK-NEXT: |-VarDecl {{.*}} PF 'float *'
// CHECK-NEXT: |-VarDecl {{.*}} PD 'double *'
// CHECK-NEXT: |-VarDecl {{.*}} CPBFalse 'const bool *'
// CHECK-NEXT: |-VarDecl {{.*}} CPBTrue 'const bool *'
// CHECK-NEXT: |-VarDecl {{.*}} CPCA 'const char *'
// CHECK-NEXT: |-VarDecl {{.*}} CPCZ 'const char *'
// CHECK-NEXT: |-VarDecl {{.*}} CPSCa 'const signed char *'
// CHECK-NEXT: |-VarDecl {{.*}} CPSCz 'const signed char *'
// CHECK-NEXT: |-VarDecl {{.*}} CPUC24 'const unsigned char *'
// CHECK-NEXT: |-VarDecl {{.*}} CPUC42 'const unsigned char *'
// CHECK-NEXT: |-VarDecl {{.*}} CPC16_24 'const char16_t *'
// CHECK-NEXT: |-VarDecl {{.*}} CPC32_42 'const char32_t *'
// CHECK-NEXT: |-VarDecl {{.*}} CPWC1 'const wchar_t *'
// CHECK-NEXT: |-VarDecl {{.*}} CPWCP 'const wchar_t *'
// CHECK-NEXT: |-VarDecl {{.*}} CPSMax 'const short *'
// CHECK-NEXT: |-VarDecl {{.*}} CPSMin 'const short *'
// CHECK-NEXT: |-VarDecl {{.*}} CPUSMax 'const unsigned short *'
// CHECK-NEXT: |-VarDecl {{.*}} CPUSMin 'const unsigned short *'
// CHECK-NEXT: |-VarDecl {{.*}} CPIMax 'const int *'
// CHECK-NEXT: |-VarDecl {{.*}} CPIMin 'const int *'
// CHECK-NEXT: |-VarDecl {{.*}} CPUIMax 'const unsigned int *'
// CHECK-NEXT: |-VarDecl {{.*}} CPUIMin 'const unsigned int *'
// CHECK-NEXT: |-VarDecl {{.*}} CPLMax 'const long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPLMin 'const long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPULMax 'const unsigned long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPULMin 'const unsigned long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPLLMax 'const long long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPLLMin 'const long long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPULLMax 'const unsigned long long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPULLMin 'const unsigned long long *'
// CHECK-NEXT: |-VarDecl {{.*}} CPF 'const float *'
// CHECK-NEXT: |-VarDecl {{.*}} CPD 'const double *'
// CHECK-NEXT: |-VarDecl {{.*}} RBFalse 'bool &'
// CHECK-NEXT: |-VarDecl {{.*}} RBTrue 'bool &'
// CHECK-NEXT: |-VarDecl {{.*}} RCA 'char &'
// CHECK-NEXT: |-VarDecl {{.*}} RCZ 'char &'
// CHECK-NEXT: |-VarDecl {{.*}} RSCa 'signed char &'
// CHECK-NEXT: |-VarDecl {{.*}} RSCz 'signed char &'
// CHECK-NEXT: |-VarDecl {{.*}} RUC24 'unsigned char &'
// CHECK-NEXT: |-VarDecl {{.*}} RUC42 'unsigned char &'
// CHECK-NEXT: |-VarDecl {{.*}} RSMax 'short &'
// CHECK-NEXT: |-VarDecl {{.*}} RSMin 'short &'
// CHECK-NEXT: |-VarDecl {{.*}} RUSMax 'unsigned short &'
// CHECK-NEXT: |-VarDecl {{.*}} RUSMin 'unsigned short &'
// CHECK-NEXT: |-VarDecl {{.*}} RIMax 'int &'
// CHECK-NEXT: |-VarDecl {{.*}} RIMin 'int &'
// CHECK-NEXT: |-VarDecl {{.*}} RUIMax 'unsigned int &'
// CHECK-NEXT: |-VarDecl {{.*}} RUIMin 'unsigned int &'
// CHECK-NEXT: |-VarDecl {{.*}} RLMax 'long &'
// CHECK-NEXT: |-VarDecl {{.*}} RLMin 'long &'
// CHECK-NEXT: |-VarDecl {{.*}} RULMax 'unsigned long &'
// CHECK-NEXT: |-VarDecl {{.*}} RULMin 'unsigned long &'
// CHECK-NEXT: |-VarDecl {{.*}} RLLMax 'long long &'
// CHECK-NEXT: |-VarDecl {{.*}} RLLMin 'long long &'
// CHECK-NEXT: |-VarDecl {{.*}} RULLMax 'unsigned long long &'
// CHECK-NEXT: |-VarDecl {{.*}} RULLMin 'unsigned long long &'
// CHECK-NEXT: |-VarDecl {{.*}} RF 'float &'
// CHECK-NEXT: |-VarDecl {{.*}} RD 'double &'
// CHECK-NEXT: |-VarDecl {{.*}} CRBFalse 'const bool &'
// CHECK-NEXT: |-VarDecl {{.*}} CRBTrue 'const bool &'
// CHECK-NEXT: |-VarDecl {{.*}} CRCA 'const char &'
// CHECK-NEXT: |-VarDecl {{.*}} CRCZ 'const char &'
// CHECK-NEXT: |-VarDecl {{.*}} CRSCa 'const signed char &'
// CHECK-NEXT: |-VarDecl {{.*}} CRSCz 'const signed char &'
// CHECK-NEXT: |-VarDecl {{.*}} CRUC24 'const unsigned char &'
// CHECK-NEXT: |-VarDecl {{.*}} CRUC42 'const unsigned char &'
// CHECK-NEXT: |-VarDecl {{.*}} CRSMax 'const short &'
// CHECK-NEXT: |-VarDecl {{.*}} CRSMin 'const short &'
// CHECK-NEXT: |-VarDecl {{.*}} CRUSMax 'const unsigned short &'
// CHECK-NEXT: |-VarDecl {{.*}} CRUSMin 'const unsigned short &'
// CHECK-NEXT: |-VarDecl {{.*}} CRIMax 'const int &'
// CHECK-NEXT: |-VarDecl {{.*}} CRIMin 'const int &'
// CHECK-NEXT: |-VarDecl {{.*}} CRUIMax 'const unsigned int &'
// CHECK-NEXT: |-VarDecl {{.*}} CRUIMin 'const unsigned int &'
// CHECK-NEXT: |-VarDecl {{.*}} CRLMax 'const long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRLMin 'const long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRULMax 'const unsigned long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRULMin 'const unsigned long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRLLMax 'const long long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRLLMin 'const long long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRULLMax 'const unsigned long long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRULLMin 'const unsigned long long &'
// CHECK-NEXT: |-VarDecl {{.*}} CRF 'const float &'
// CHECK-NEXT: |-VarDecl {{.*}} CRD 'const double &'
// CHECK-NEXT: |-VarDecl {{.*}} RC16_24 'char16_t &'
// CHECK-NEXT: |-VarDecl {{.*}} RC32_42 'char32_t &'
// CHECK-NEXT: |-VarDecl {{.*}} RWC1 'wchar_t &'
// CHECK-NEXT: |-VarDecl {{.*}} RWCP 'wchar_t &'
// CHECK-NEXT: |-VarDecl {{.*}} CRC16_24 'const char16_t &'
// CHECK-NEXT: |-VarDecl {{.*}} CRC32_42 'const char32_t &'
// CHECK-NEXT: |-VarDecl {{.*}} CRWC1 'const wchar_t &'
// CHECK-NEXT: `-VarDecl {{.*}} CRWCP 'const wchar_t &'

// CHECK: (lldb) quit

int main(int argc, char **argv) {
  return CIMax;
}

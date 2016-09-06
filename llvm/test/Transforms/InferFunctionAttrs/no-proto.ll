; RUN: opt < %s -mtriple=x86_64-unknown-linux -inferattrs -S | FileCheck %s
; RUN: opt < %s -mtriple=x86_64-apple-macosx10.8.0 -inferattrs -S | FileCheck %s

; Check that we don't modify libc functions with invalid prototypes.

; CHECK: declare void @__cospi(...)
declare void @__cospi(...)

; CHECK: declare void @__cospif(...)
declare void @__cospif(...)

; CHECK: declare void @__sinpi(...)
declare void @__sinpi(...)

; CHECK: declare void @__sinpif(...)
declare void @__sinpif(...)

; CHECK: declare void @abs(...)
declare void @abs(...)

; CHECK: declare void @access(...)
declare void @access(...)

; CHECK: declare void @acos(...)
declare void @acos(...)

; CHECK: declare void @acosf(...)
declare void @acosf(...)

; CHECK: declare void @acosh(...)
declare void @acosh(...)

; CHECK: declare void @acoshf(...)
declare void @acoshf(...)

; CHECK: declare void @acoshl(...)
declare void @acoshl(...)

; CHECK: declare void @acosl(...)
declare void @acosl(...)

; CHECK: declare void @asin(...)
declare void @asin(...)

; CHECK: declare void @asinf(...)
declare void @asinf(...)

; CHECK: declare void @asinh(...)
declare void @asinh(...)

; CHECK: declare void @asinhf(...)
declare void @asinhf(...)

; CHECK: declare void @asinhl(...)
declare void @asinhl(...)

; CHECK: declare void @asinl(...)
declare void @asinl(...)

; CHECK: declare void @atan(...)
declare void @atan(...)

; CHECK: declare void @atan2(...)
declare void @atan2(...)

; CHECK: declare void @atan2f(...)
declare void @atan2f(...)

; CHECK: declare void @atan2l(...)
declare void @atan2l(...)

; CHECK: declare void @atanf(...)
declare void @atanf(...)

; CHECK: declare void @atanh(...)
declare void @atanh(...)

; CHECK: declare void @atanhf(...)
declare void @atanhf(...)

; CHECK: declare void @atanhl(...)
declare void @atanhl(...)

; CHECK: declare void @atanl(...)
declare void @atanl(...)

; CHECK: declare void @atof(...)
declare void @atof(...)

; CHECK: declare void @atoi(...)
declare void @atoi(...)

; CHECK: declare void @atol(...)
declare void @atol(...)

; CHECK: declare void @atoll(...)
declare void @atoll(...)

; CHECK: declare void @bcmp(...)
declare void @bcmp(...)

; CHECK: declare void @bcopy(...)
declare void @bcopy(...)

; CHECK: declare void @bzero(...)
declare void @bzero(...)

; CHECK: declare void @calloc(...)
declare void @calloc(...)

; CHECK: declare void @cbrt(...)
declare void @cbrt(...)

; CHECK: declare void @cbrtf(...)
declare void @cbrtf(...)

; CHECK: declare void @cbrtl(...)
declare void @cbrtl(...)

; CHECK: declare void @ceil(...)
declare void @ceil(...)

; CHECK: declare void @ceilf(...)
declare void @ceilf(...)

; CHECK: declare void @ceill(...)
declare void @ceill(...)

; CHECK: declare void @chmod(...)
declare void @chmod(...)

; CHECK: declare void @chown(...)
declare void @chown(...)

; CHECK: declare void @clearerr(...)
declare void @clearerr(...)

; CHECK: declare void @closedir(...)
declare void @closedir(...)

; CHECK: declare void @copysign(...)
declare void @copysign(...)

; CHECK: declare void @copysignf(...)
declare void @copysignf(...)

; CHECK: declare void @copysignl(...)
declare void @copysignl(...)

; CHECK: declare void @cos(...)
declare void @cos(...)

; CHECK: declare void @cosf(...)
declare void @cosf(...)

; CHECK: declare void @cosh(...)
declare void @cosh(...)

; CHECK: declare void @coshf(...)
declare void @coshf(...)

; CHECK: declare void @coshl(...)
declare void @coshl(...)

; CHECK: declare void @cosl(...)
declare void @cosl(...)

; CHECK: declare void @ctermid(...)
declare void @ctermid(...)

; CHECK: declare void @exp(...)
declare void @exp(...)

; CHECK: declare void @exp2(...)
declare void @exp2(...)

; CHECK: declare void @exp2f(...)
declare void @exp2f(...)

; CHECK: declare void @exp2l(...)
declare void @exp2l(...)

; CHECK: declare void @expf(...)
declare void @expf(...)

; CHECK: declare void @expl(...)
declare void @expl(...)

; CHECK: declare void @expm1(...)
declare void @expm1(...)

; CHECK: declare void @expm1f(...)
declare void @expm1f(...)

; CHECK: declare void @expm1l(...)
declare void @expm1l(...)

; CHECK: declare void @fabs(...)
declare void @fabs(...)

; CHECK: declare void @fabsf(...)
declare void @fabsf(...)

; CHECK: declare void @fabsl(...)
declare void @fabsl(...)

; CHECK: declare void @fclose(...)
declare void @fclose(...)

; CHECK: declare void @fdopen(...)
declare void @fdopen(...)

; CHECK: declare void @feof(...)
declare void @feof(...)

; CHECK: declare void @ferror(...)
declare void @ferror(...)

; CHECK: declare void @fflush(...)
declare void @fflush(...)

; CHECK: declare void @ffs(...)
declare void @ffs(...)

; CHECK: declare void @ffsl(...)
declare void @ffsl(...)

; CHECK: declare void @ffsll(...)
declare void @ffsll(...)

; CHECK: declare void @fgetc(...)
declare void @fgetc(...)

; CHECK: declare void @fgetpos(...)
declare void @fgetpos(...)

; CHECK: declare void @fgets(...)
declare void @fgets(...)

; CHECK: declare void @fileno(...)
declare void @fileno(...)

; CHECK: declare void @flockfile(...)
declare void @flockfile(...)

; CHECK: declare void @floor(...)
declare void @floor(...)

; CHECK: declare void @floorf(...)
declare void @floorf(...)

; CHECK: declare void @floorl(...)
declare void @floorl(...)

; CHECK: declare void @fls(...)
declare void @fls(...)

; CHECK: declare void @flsl(...)
declare void @flsl(...)

; CHECK: declare void @flsll(...)
declare void @flsll(...)

; CHECK: declare void @fmax(...)
declare void @fmax(...)

; CHECK: declare void @fmaxf(...)
declare void @fmaxf(...)

; CHECK: declare void @fmaxl(...)
declare void @fmaxl(...)

; CHECK: declare void @fmin(...)
declare void @fmin(...)

; CHECK: declare void @fminf(...)
declare void @fminf(...)

; CHECK: declare void @fminl(...)
declare void @fminl(...)

; CHECK: declare void @fmod(...)
declare void @fmod(...)

; CHECK: declare void @fmodf(...)
declare void @fmodf(...)

; CHECK: declare void @fmodl(...)
declare void @fmodl(...)

; CHECK: declare void @fopen(...)
declare void @fopen(...)

; CHECK: declare void @fprintf(...)
declare void @fprintf(...)

; CHECK: declare void @fputc(...)
declare void @fputc(...)

; CHECK: declare void @fputs(...)
declare void @fputs(...)

; CHECK: declare void @fread(...)
declare void @fread(...)

; CHECK: declare void @free(...)
declare void @free(...)

; CHECK: declare void @frexp(...)
declare void @frexp(...)

; CHECK: declare void @frexpf(...)
declare void @frexpf(...)

; CHECK: declare void @frexpl(...)
declare void @frexpl(...)

; CHECK: declare void @fscanf(...)
declare void @fscanf(...)

; CHECK: declare void @fseek(...)
declare void @fseek(...)

; CHECK: declare void @fseeko(...)
declare void @fseeko(...)

; CHECK: declare void @fseeko64(...)
declare void @fseeko64(...)

; CHECK: declare void @fsetpos(...)
declare void @fsetpos(...)

; CHECK: declare void @fstat(...)
declare void @fstat(...)

; CHECK: declare void @fstat64(...)
declare void @fstat64(...)

; CHECK: declare void @fstatvfs(...)
declare void @fstatvfs(...)

; CHECK: declare void @fstatvfs64(...)
declare void @fstatvfs64(...)

; CHECK: declare void @ftell(...)
declare void @ftell(...)

; CHECK: declare void @ftello(...)
declare void @ftello(...)

; CHECK: declare void @ftello64(...)
declare void @ftello64(...)

; CHECK: declare void @ftrylockfile(...)
declare void @ftrylockfile(...)

; CHECK: declare void @funlockfile(...)
declare void @funlockfile(...)

; CHECK: declare void @fwrite(...)
declare void @fwrite(...)

; CHECK: declare void @getc(...)
declare void @getc(...)

; CHECK: declare void @getc_unlocked(...)
declare void @getc_unlocked(...)

; CHECK: declare void @getchar(...)
declare void @getchar(...)

; CHECK: declare void @getenv(...)
declare void @getenv(...)

; CHECK: declare void @getitimer(...)
declare void @getitimer(...)

; CHECK: declare void @getlogin_r(...)
declare void @getlogin_r(...)

; CHECK: declare void @getpwnam(...)
declare void @getpwnam(...)

; CHECK: declare void @gets(...)
declare void @gets(...)

; CHECK: declare void @gettimeofday(...)
declare void @gettimeofday(...)

; CHECK: declare void @isascii(...)
declare void @isascii(...)

; CHECK: declare void @isdigit(...)
declare void @isdigit(...)

; CHECK: declare void @labs(...)
declare void @labs(...)

; CHECK: declare void @lchown(...)
declare void @lchown(...)

; CHECK: declare void @ldexp(...)
declare void @ldexp(...)

; CHECK: declare void @ldexpf(...)
declare void @ldexpf(...)

; CHECK: declare void @ldexpl(...)
declare void @ldexpl(...)

; CHECK: declare void @llabs(...)
declare void @llabs(...)

; CHECK: declare void @log(...)
declare void @log(...)

; CHECK: declare void @log10(...)
declare void @log10(...)

; CHECK: declare void @log10f(...)
declare void @log10f(...)

; CHECK: declare void @log10l(...)
declare void @log10l(...)

; CHECK: declare void @log1p(...)
declare void @log1p(...)

; CHECK: declare void @log1pf(...)
declare void @log1pf(...)

; CHECK: declare void @log1pl(...)
declare void @log1pl(...)

; CHECK: declare void @log2(...)
declare void @log2(...)

; CHECK: declare void @log2f(...)
declare void @log2f(...)

; CHECK: declare void @log2l(...)
declare void @log2l(...)

; CHECK: declare void @logb(...)
declare void @logb(...)

; CHECK: declare void @logbf(...)
declare void @logbf(...)

; CHECK: declare void @logbl(...)
declare void @logbl(...)

; CHECK: declare void @logf(...)
declare void @logf(...)

; CHECK: declare void @logl(...)
declare void @logl(...)

; CHECK: declare void @lstat(...)
declare void @lstat(...)

; CHECK: declare void @lstat64(...)
declare void @lstat64(...)

; CHECK: declare void @malloc(...)
declare void @malloc(...)

; CHECK: declare void @memalign(...)
declare void @memalign(...)

; CHECK: declare void @memccpy(...)
declare void @memccpy(...)

; CHECK: declare void @memchr(...)
declare void @memchr(...)

; CHECK: declare void @memcmp(...)
declare void @memcmp(...)

; CHECK: declare void @memcpy(...)
declare void @memcpy(...)

; CHECK: declare void @mempcpy(...)
declare void @mempcpy(...)

; CHECK: declare void @memmove(...)
declare void @memmove(...)

; CHECK: declare void @memset(...)
declare void @memset(...)

; CHECK: declare void @memset_pattern16(...)
declare void @memset_pattern16(...)

; CHECK: declare void @mkdir(...)
declare void @mkdir(...)

; CHECK: declare void @mktime(...)
declare void @mktime(...)

; CHECK: declare void @modf(...)
declare void @modf(...)

; CHECK: declare void @modff(...)
declare void @modff(...)

; CHECK: declare void @modfl(...)
declare void @modfl(...)

; CHECK: declare void @nearbyint(...)
declare void @nearbyint(...)

; CHECK: declare void @nearbyintf(...)
declare void @nearbyintf(...)

; CHECK: declare void @nearbyintl(...)
declare void @nearbyintl(...)

; CHECK: declare void @open(...)
declare void @open(...)

; CHECK: declare void @open64(...)
declare void @open64(...)

; CHECK: declare void @opendir(...)
declare void @opendir(...)

; CHECK: declare void @pclose(...)
declare void @pclose(...)

; CHECK: declare void @perror(...)
declare void @perror(...)

; CHECK: declare void @popen(...)
declare void @popen(...)

; CHECK: declare void @posix_memalign(...)
declare void @posix_memalign(...)

; CHECK: declare void @pow(...)
declare void @pow(...)

; CHECK: declare void @powf(...)
declare void @powf(...)

; CHECK: declare void @powl(...)
declare void @powl(...)

; CHECK: declare void @pread(...)
declare void @pread(...)

; CHECK: declare void @printf(...)
declare void @printf(...)

; CHECK: declare void @putc(...)
declare void @putc(...)

; CHECK: declare void @putchar(...)
declare void @putchar(...)

; CHECK: declare void @puts(...)
declare void @puts(...)

; CHECK: declare void @pwrite(...)
declare void @pwrite(...)

; CHECK: declare void @qsort(...)
declare void @qsort(...)

; CHECK: declare void @read(...)
declare void @read(...)

; CHECK: declare void @readlink(...)
declare void @readlink(...)

; CHECK: declare void @realloc(...)
declare void @realloc(...)

; CHECK: declare void @reallocf(...)
declare void @reallocf(...)

; CHECK: declare void @realpath(...)
declare void @realpath(...)

; CHECK: declare void @remove(...)
declare void @remove(...)

; CHECK: declare void @rename(...)
declare void @rename(...)

; CHECK: declare void @rewind(...)
declare void @rewind(...)

; CHECK: declare void @rint(...)
declare void @rint(...)

; CHECK: declare void @rintf(...)
declare void @rintf(...)

; CHECK: declare void @rintl(...)
declare void @rintl(...)

; CHECK: declare void @rmdir(...)
declare void @rmdir(...)

; CHECK: declare void @round(...)
declare void @round(...)

; CHECK: declare void @roundf(...)
declare void @roundf(...)

; CHECK: declare void @roundl(...)
declare void @roundl(...)

; CHECK: declare void @scanf(...)
declare void @scanf(...)

; CHECK: declare void @setbuf(...)
declare void @setbuf(...)

; CHECK: declare void @setitimer(...)
declare void @setitimer(...)

; CHECK: declare void @setvbuf(...)
declare void @setvbuf(...)

; CHECK: declare void @sin(...)
declare void @sin(...)

; CHECK: declare void @sinf(...)
declare void @sinf(...)

; CHECK: declare void @sinh(...)
declare void @sinh(...)

; CHECK: declare void @sinhf(...)
declare void @sinhf(...)

; CHECK: declare void @sinhl(...)
declare void @sinhl(...)

; CHECK: declare void @sinl(...)
declare void @sinl(...)

; CHECK: declare void @snprintf(...)
declare void @snprintf(...)

; CHECK: declare void @sprintf(...)
declare void @sprintf(...)

; CHECK: declare void @sqrt(...)
declare void @sqrt(...)

; CHECK: declare void @sqrtf(...)
declare void @sqrtf(...)

; CHECK: declare void @sqrtl(...)
declare void @sqrtl(...)

; CHECK: declare void @sscanf(...)
declare void @sscanf(...)

; CHECK: declare void @stat(...)
declare void @stat(...)

; CHECK: declare void @stat64(...)
declare void @stat64(...)

; CHECK: declare void @statvfs(...)
declare void @statvfs(...)

; CHECK: declare void @statvfs64(...)
declare void @statvfs64(...)

; CHECK: declare void @stpcpy(...)
declare void @stpcpy(...)

; CHECK: declare void @stpncpy(...)
declare void @stpncpy(...)

; CHECK: declare void @strcasecmp(...)
declare void @strcasecmp(...)

; CHECK: declare void @strcat(...)
declare void @strcat(...)

; CHECK: declare void @strchr(...)
declare void @strchr(...)

; CHECK: declare void @strcmp(...)
declare void @strcmp(...)

; CHECK: declare void @strcoll(...)
declare void @strcoll(...)

; CHECK: declare void @strcpy(...)
declare void @strcpy(...)

; CHECK: declare void @strcspn(...)
declare void @strcspn(...)

; CHECK: declare void @strdup(...)
declare void @strdup(...)

; CHECK: declare void @strlen(...)
declare void @strlen(...)

; CHECK: declare void @strncasecmp(...)
declare void @strncasecmp(...)

; CHECK: declare void @strncat(...)
declare void @strncat(...)

; CHECK: declare void @strncmp(...)
declare void @strncmp(...)

; CHECK: declare void @strncpy(...)
declare void @strncpy(...)

; CHECK: declare void @strndup(...)
declare void @strndup(...)

; CHECK: declare void @strnlen(...)
declare void @strnlen(...)

; CHECK: declare void @strpbrk(...)
declare void @strpbrk(...)

; CHECK: declare void @strrchr(...)
declare void @strrchr(...)

; CHECK: declare void @strspn(...)
declare void @strspn(...)

; CHECK: declare void @strstr(...)
declare void @strstr(...)

; CHECK: declare void @strtod(...)
declare void @strtod(...)

; CHECK: declare void @strtof(...)
declare void @strtof(...)

; CHECK: declare void @strtok(...)
declare void @strtok(...)

; CHECK: declare void @strtok_r(...)
declare void @strtok_r(...)

; CHECK: declare void @strtol(...)
declare void @strtol(...)

; CHECK: declare void @strtold(...)
declare void @strtold(...)

; CHECK: declare void @strtoll(...)
declare void @strtoll(...)

; CHECK: declare void @strtoul(...)
declare void @strtoul(...)

; CHECK: declare void @strtoull(...)
declare void @strtoull(...)

; CHECK: declare void @strxfrm(...)
declare void @strxfrm(...)

; CHECK: declare void @system(...)
declare void @system(...)

; CHECK: declare void @tan(...)
declare void @tan(...)

; CHECK: declare void @tanf(...)
declare void @tanf(...)

; CHECK: declare void @tanh(...)
declare void @tanh(...)

; CHECK: declare void @tanhf(...)
declare void @tanhf(...)

; CHECK: declare void @tanhl(...)
declare void @tanhl(...)

; CHECK: declare void @tanl(...)
declare void @tanl(...)

; CHECK: declare void @times(...)
declare void @times(...)

; CHECK: declare void @tmpfile(...)
declare void @tmpfile(...)

; CHECK: declare void @tmpfile64(...)
declare void @tmpfile64(...)

; CHECK: declare void @toascii(...)
declare void @toascii(...)

; CHECK: declare void @trunc(...)
declare void @trunc(...)

; CHECK: declare void @truncf(...)
declare void @truncf(...)

; CHECK: declare void @truncl(...)
declare void @truncl(...)

; CHECK: declare void @uname(...)
declare void @uname(...)

; CHECK: declare void @ungetc(...)
declare void @ungetc(...)

; CHECK: declare void @unlink(...)
declare void @unlink(...)

; CHECK: declare void @unsetenv(...)
declare void @unsetenv(...)

; CHECK: declare void @utime(...)
declare void @utime(...)

; CHECK: declare void @utimes(...)
declare void @utimes(...)

; CHECK: declare void @valloc(...)
declare void @valloc(...)

; CHECK: declare void @vfprintf(...)
declare void @vfprintf(...)

; CHECK: declare void @vfscanf(...)
declare void @vfscanf(...)

; CHECK: declare void @vprintf(...)
declare void @vprintf(...)

; CHECK: declare void @vscanf(...)
declare void @vscanf(...)

; CHECK: declare void @vsnprintf(...)
declare void @vsnprintf(...)

; CHECK: declare void @vsprintf(...)
declare void @vsprintf(...)

; CHECK: declare void @vsscanf(...)
declare void @vsscanf(...)

; CHECK: declare void @write(...)
declare void @write(...)

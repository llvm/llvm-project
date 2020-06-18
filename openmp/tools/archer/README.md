<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#org8ca70b5">1. License</a></li>
<li><a href="#orgc6a2b10">2. Introduction</a></li>
<li><a href="#org9a459f1">3. Installation</a></li>
<li><a href="#orgb820ad0">4. Usage</a>
<ul>
<li><a href="#org213ff1a">4.1. How to compile</a></li>
<li><a href="#org110062c">4.2. Runtime Flags</a></li>
</ul>
</li>
<li><a href="#org73e58a9">5. Example</a></li>
<li><a href="#orgcc38a36">6. Contacts and Support</a></li>
</ul>
</div>
</div>


<a id="org8ca70b5"></a>

# License

Archer is distributed under the terms of the Apache License.

Please see LICENSE.txt for usage terms.

LLNL-CODE-773957

<a id="orgc6a2b10"></a>

# Introduction

**Archer** is an OMPT tool which annotates OpenMP synchronization semantics for data race
detection.
This avoids false alerts in data race detection.
Archer is automatically loaded for OpenMP applications which are compiled
with ThreadSanitizer option.

<a id="org9a459f1"></a>

# Build Archer within Clang/LLVM

This distribution of Archer is automatically built with the OpenMP runtime
and automatically loaded by the OpenMP runtime.

<a id="orgb820ad0"></a>

# Usage


<a id="org213ff1a"></a>

## How to compile

To use archer, compile the application with the extra flag
`-fsanitize=thread`:

    clang -O3 -g -fopenmp -fsanitize=thread app.c
    clang++ -O3 -g -fopenmp -fsanitize=thread app.cpp

To compile Fortran applications, compile with gfortran, link with clang:

    gfortran -g -c -fopenmp -fsanitize=thread app.f
    clang -fopenmp -fsanitize=thread app.o -lgfortran


<a id="org110062c"></a>

## Runtime Flags

TSan runtime flags are passed via **TSAN&#95;OPTIONS** environment variable,
we highly recommend the following option to avoid false alerts for the
OpenMP or MPI runtime implementation:

    export TSAN_OPTIONS="ignore_noninstrumented_modules=1"


Runtime flags are passed via **ARCHER&#95;OPTIONS** environment variable,
different flags are separated by spaces, e.g.:

    ARCHER_OPTIONS="flush_shadow=1" ./myprogram

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-right" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Flag Name</th>
<th scope="col" class="org-right">Default value</th>
<th scope="col" class="org-left">Description</th>
</tr>
</thead>

<tbody>
<tr>
<td class="org-left">flush&#95;shadow</td>
<td class="org-right">0</td>
<td class="org-left">Flush shadow memory at the end of an outer OpenMP parallel region. Our experiments show that this can reduce memory overhead by ~30% and runtime overhead by ~10%. This flag is useful for large OpenMP applications that typically require large amounts of memory, causing out-of-memory exceptions when checked by Archer.</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">print&#95;max&#95;rss</td>
<td class="org-right">0</td>
<td class="org-left">Print the RSS memory peak at the end of the execution.</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">verbose</td>
<td class="org-right">0</td>
<td class="org-left">Print startup information.</td>
</tr>
</tbody>

<tbody>
<tr>
<td class="org-left">enable</td>
<td class="org-right">1</td>
<td class="org-left">Use Archer runtime library during execution.</td>
</tr>
</tbody>
</table>


<a id="org73e58a9"></a>

# Example

Let us take the program below and follow the steps to compile and
check the program for data races.

Suppose our program is called *myprogram.c*:

     1  #include <stdio.h>
     2
     3  #define N 1000
     4
     5  int main (int argc, char **argv)
     6  {
     7    int a[N];
     8
     9  #pragma omp parallel for
    10    for (int i = 0; i < N - 1; i++) {
    11      a[i] = a[i + 1];
    12    }
    13  }

We compile the program as follow:

    clang -fsanitize=thread -fopenmp -g myprogram.c -o myprogram

Now we can run the program with the following commands:

    export OMP_NUM_THREADS=2
    ./myprogram

Archer will output a report in case it finds data races. In our case
the report will look as follow:

    ==================
    WARNING: ThreadSanitizer: data race (pid=13641)
      Read of size 4 at 0x7fff79a01170 by main thread:
        #0 .omp_outlined. myprogram.c:11:12 (myprogram+0x00000049b5a2)
        #1 __kmp_invoke_microtask <null> (libomp.so+0x000000077842)
        #2 __libc_start_main /build/glibc-t3gR2i/glibc-2.23/csu/../csu/libc-start.c:291 (libc.so.6+0x00000002082f)

      Previous write of size 4 at 0x7fff79a01170 by thread T1:
        #0 .omp_outlined. myprogram.c:11:10 (myprogram+0x00000049b5d6)
        #1 __kmp_invoke_microtask <null> (libomp.so+0x000000077842)

      Location is stack of main thread.

      Thread T1 (tid=13643, running) created by main thread at:
        #0 pthread_create tsan_interceptors.cc:902:3 (myprogram+0x00000043db75)
        #1 __kmp_create_worker <null> (libomp.so+0x00000006c364)
        #2 __libc_start_main /build/glibc-t3gR2i/glibc-2.23/csu/../csu/libc-start.c:291 (libc.so.6+0x00000002082f)

    SUMMARY: ThreadSanitizer: data race myprogram.c:11:12 in .omp_outlined.
    ==================
    ThreadSanitizer: reported 1 warnings


<a id="orgcc38a36"></a>

# Contacts and Support

-   [Google group](https://groups.google.com/forum/#!forum/archer-pruner)
-   [Slack Channel](https://pruners.slack.com)

    <ul style="list-style-type:circle"> <li> For an invitation please write an email to <a href="mailto:simone@cs.utah.edu?Subject=[archer-slack] Slack Invitation" target="_top">Simone Atzeni</a> with a reason why you want to be part of the PRUNERS Slack Team. </li> </ul>
-   E-Mail Contacts:

    <ul style="list-style-type:circle"> <li> <a href="mailto:simone@cs.utah.edu?Subject=[archer-dev]%20" target="_top">Simone Atzeni</a> </li> <li> <a href="mailto:protze@itc.rwth-aachen.de?Subject=[archer-dev]%20" target="_top">Joachim Protze</a> </li> </ul>



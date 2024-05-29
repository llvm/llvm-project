# RUN: llvm-mc -triple x86_64 -show-encoding -x86-asm-syntax=intel -output-asm-variant=1 %s | FileCheck %s

# CHECK: adcx	r17d, r16d
# CHECK: encoding: [0x62,0xec,0x7d,0x08,0x66,0xc8]
         adcx	r17d, r16d
# CHECK: adcx	r18d, r17d, r16d
# CHECK: encoding: [0x62,0xec,0x6d,0x10,0x66,0xc8]
         adcx	r18d, r17d, r16d
# CHECK: adcx	r17, r16
# CHECK: encoding: [0x62,0xec,0xfd,0x08,0x66,0xc8]
         adcx	r17, r16
# CHECK: adcx	r18, r17, r16
# CHECK: encoding: [0x62,0xec,0xed,0x10,0x66,0xc8]
         adcx	r18, r17, r16
# CHECK: adcx	r17d, dword ptr [r16]
# CHECK: encoding: [0x62,0xec,0x7d,0x08,0x66,0x08]
         adcx	r17d, dword ptr [r16]
# CHECK: adcx	r18d, r17d, dword ptr [r16]
# CHECK: encoding: [0x62,0xec,0x6d,0x10,0x66,0x08]
         adcx	r18d, r17d, dword ptr [r16]
# CHECK: adcx	r17, qword ptr [r16]
# CHECK: encoding: [0x62,0xec,0xfd,0x08,0x66,0x08]
         adcx	r17, qword ptr [r16]
# CHECK: adcx	r18, r17, qword ptr [r16]
# CHECK: encoding: [0x62,0xec,0xed,0x10,0x66,0x08]
         adcx	r18, r17, qword ptr [r16]
# CHECK: adox	r17d, r16d
# CHECK: encoding: [0x62,0xec,0x7e,0x08,0x66,0xc8]
         adox	r17d, r16d
# CHECK: adox	r18d, r17d, r16d
# CHECK: encoding: [0x62,0xec,0x6e,0x10,0x66,0xc8]
         adox	r18d, r17d, r16d
# CHECK: adox	r17, r16
# CHECK: encoding: [0x62,0xec,0xfe,0x08,0x66,0xc8]
         adox	r17, r16
# CHECK: adox	r18, r17, r16
# CHECK: encoding: [0x62,0xec,0xee,0x10,0x66,0xc8]
         adox	r18, r17, r16
# CHECK: adox	r17d, dword ptr [r16]
# CHECK: encoding: [0x62,0xec,0x7e,0x08,0x66,0x08]
         adox	r17d, dword ptr [r16]
# CHECK: adox	r18d, r17d, dword ptr [r16]
# CHECK: encoding: [0x62,0xec,0x6e,0x10,0x66,0x08]
         adox	r18d, r17d, dword ptr [r16]
# CHECK: adox	r17, qword ptr [r16]
# CHECK: encoding: [0x62,0xec,0xfe,0x08,0x66,0x08]
         adox	r17, qword ptr [r16]
# CHECK: adox	r18, r17, qword ptr [r16]
# CHECK: encoding: [0x62,0xec,0xee,0x10,0x66,0x08]
         adox	r18, r17, qword ptr [r16]

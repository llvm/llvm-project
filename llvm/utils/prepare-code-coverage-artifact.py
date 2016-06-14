#!/usr/bin/env python

'''Prepare a code coverage artifact.

- Collate raw profiles into one indexed profile.
- Delete the raw profiles.
- Copy the coverage mappings in the binaries directory.
'''

import argparse
import glob
import os
import subprocess
import sys

def merge_raw_profiles(host_llvm_profdata, profile_data_dir):
    print ':: Merging raw profiles...',
    sys.stdout.flush()
    raw_profiles = glob.glob(os.path.join(profile_data_dir, '*.profraw'))
    manifest_path = os.path.join(profile_data_dir, 'profiles.manifest')
    profdata_path = os.path.join(profile_data_dir, 'Coverage.profdata')
    with open(manifest_path, 'w') as manifest:
        manifest.write('\n'.join(raw_profiles))
    subprocess.check_call([host_llvm_profdata, 'merge', '-sparse', '-f',
                           manifest_path, '-o', profdata_path])
    for raw_profile in raw_profiles:
        os.remove(raw_profile)
    print 'Done!'

def extract_covmappings(host_llvm_cov, profile_data_dir, llvm_bin_dir):
    print ':: Extracting covmappings...',
    sys.stdout.flush()
    for prog in os.listdir(llvm_bin_dir):
        if prog == 'llvm-lit':
            continue
        covmapping_path = os.path.join(profile_data_dir,
                                       os.path.basename(prog) + '.covmapping')
        subprocess.check_call([host_llvm_cov, 'convert-for-testing',
                               os.path.join(llvm_bin_dir, prog), '-o',
                               covmapping_path])
    print 'Done!'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host_llvm_profdata', help='Path to llvm-profdata')
    parser.add_argument('host_llvm_cov', help='Path to llvm-cov')
    parser.add_argument('profile_data_dir',
                       help='Path to the directory containing the raw profiles')
    parser.add_argument('llvm_bin_dir',
                       help='Path to the directory containing llvm binaries')
    args = parser.parse_args()

    merge_raw_profiles(args.host_llvm_profdata, args.profile_data_dir)
    extract_covmappings(args.host_llvm_cov, args.profile_data_dir,
                        args.llvm_bin_dir)

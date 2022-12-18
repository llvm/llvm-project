#!/usr/bin/env python3

import subprocess
import glob
import json


def test_1():
    # --- compile code ---
    cmd = ["make"]
    try:
        cmd_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    # cmd = ["make run"]
    # try:
    #     cmd_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    # except subprocess.CalledProcessError as e:
    #     print(e.output)
    #     exit()
    #
    # file_name = glob.glob('.fAC_logs/fAC_*.json')[0]
    # data = json.load(file_name)
    # assert len(data["FP32"]) == 20
    # assert len(data["FP64"]) == 21

    # --- Cleanup ---
    # cmd = ["make clean"]
    # try:
    #     cmd_output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    # except subprocess.CalledProcessError as e:
    #     print(e.output)
    #     exit()


test_1()

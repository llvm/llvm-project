import sys
import os.path
import traceback

if __name__ == "__main__":
    try:
        sys.path.append(os.path.dirname(__file__))

        from ompd import ompd

        ompd.main()
        print('OMPD GDB support loaded')
    except Exception as e:
        traceback.print_exc()
        print('Error: OMPD support could not be loaded', e)
